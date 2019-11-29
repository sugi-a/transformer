import argparse, sys, os, time
import tensorflow as tf
import numpy as np
from logging import getLogger, INFO, DEBUG, basicConfig
logger = getLogger(__name__)

from .model import *
from .utils import *
from . import dataprocessing

class Inference:
    def __init__(self, model_dir, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=4, batch_capacity=None, sampling_method=None):

        # Model's working directory
        self.model_dir = model_dir

        # log dir
        self.logdir = self.model_dir + '/log'

        # Load configuration
        self.config = {'model_dir': model_dir}
        with open(self.model_dir + '/model_config.py', 'r') as f:
            exec(f.read(), self.config)
        
        self.params = self.config["params"]
        params = self.params

        # Select the method to convert IDs to tokens
        if 'IDs2text' in self.config:
            self.IDs2text = self.config["IDs2text"]
            logger.debug('Inference using the custom IDs2text method')
        else:
            self.vocab = dataprocessing.Vocabulary(
                params["vocab"]["target_dict"],
                UNK_ID= params["vocab"]["UNK_ID"],
                EOS_ID= params["vocab"]["EOS_ID"],
                PAD_ID= params["vocab"]["PAD_ID"])
            self.IDs2text = self.vocab.IDs2text
            logger.debug('Inference using the default IDs2text method')
        
        # Computing options
        self.n_cpu_cores = n_cpu_cores
        self.n_gpus = n_gpus
        self.batch_capacity = batch_capacity or 8192 * self.n_gpus

        # Checkpoint
        self.checkpoint = checkpoint or tf.train.latest_checkpoint(self.logdir + '/sup_checkpoint')

        # Session of this inference class. An instance is set by `self.make_session()`
        self.session = None

        # set the given model or create a new one
        if model is None:
            self.graph = graph or tf.get_default_graph()

            with self.graph.as_default():
                self.model = Transformer(params)
                if self.n_gpus == 1:
                    # Single GPU
                    with tf.device('/gpu:0'):
                        self.model.instanciate_vars()
                else:
                    # place variables in the device /cpu:0
                    with tf.device('/cpu:0'):
                        self.model.instanciate_vars()
        else:
            self.graph = model.graph
            self.model = model

        # Computation graph components
        with self.graph.as_default():
            # Placeholder for input data
            self.ph_dict = {
                'x': tf.placeholder(tf.int32, [None, None]),
                'x_len': tf.placeholder(tf.int32, [None]),
                'init_y': tf.placeholder(tf.int32, [None, None]),
                'init_y_len': tf.placeholder(tf.int32, [None])
            }

            # Splitting for data parallel computing
            self.inputs_parallel = non_even_split(
                ((self.ph_dict['x'], self.ph_dict['x_len']),
                (self.ph_dict['init_y'], self.ph_dict['init_y_len'])), self.n_gpus)
            
            # computation graph for beam search
            self.ph_beam_size = tf.placeholder(tf.int32, [])
            self.sampling_method = sampling_method
            self.op_beam_hypos_scores = self.make_op(self.fn_beam_search)

            # Computation graph for perplexity
            self.op_perplexity = self.make_op(self.fn_perplexity)

            # Computation graph for translation score
            self.ph_length_penalty = tf.placeholder(tf.float64, [])
            self.op_trans_score = self.make_op(self.fn_translation_score)

    def fn_beam_search(self, inputs):
        (x, x_len), (init_y, init_y_len) = inputs
        beam_candidates, scores = self.model.decode(
            x,
            x_len,
            self.ph_beam_size,
            return_search_results=True,
            init_y=init_y,
            init_y_len=init_y_len,
            sampling_method=self.sampling_method)
        return beam_candidates, scores

    def fn_perplexity(self, inputs):
        (x, x_len), (y, y_len) = inputs
        logits = self.model.get_logits(x, y, x_len, y_len, False)
        is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, y[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]
        perp = tf.exp(-seq_log_prob / tf.cast(y_len, tf.float32))

        return [perp]


    def fn_translation_score(self, inputs):
        (x, x_len), (y, y_len) = inputs
        logits = self.model.get_logits(x, y, x_len, y_len, False)
        is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, y[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]

        # penalty coefficient (defined in model.py)
        penalty = length_penalty(y_len, self.ph_length_penalty)

        score = seq_log_prob / penalty 
        
        return [score]

    def make_op(self, fn):
        """Create operation which computes the function specified with GPU(s) in parallel.
        Args:
            fn:
                Args: inputs = ((x, x_len), (y, y_len))
                Returns: tuple or list (ret1, ret2, ...). ret: [BATCH_SIZE, ...]
        Returns:
            list of replicated operations to be computed in parallel.
            """
        return compute_parallel(fn, self.inputs_parallel)

    def make_feed_dict(self, batch):
        return {self.ph_dict['x']: batch[0][0],
                    self.ph_dict['x_len']: batch[0][1],
                    self.ph_dict['init_y']: batch[1][0],
                    self.ph_dict['init_y_len']: batch[1][1]}


    def execute_op(self, op, batches, _feed_dict=None):
        """Evaluate `op` in the session. `op` must be created by `self.make_op`
        Args:
            op: operation created by `self.make_op`
            batches: input data
            _feed_dict: custom feed_dict
        Returns:
            Value of `op` in python list format (not numpy)"""
        assert self.session
        
        run_results = []
        start_time = time.time()
        sys.stderr.write('{} steps\r'.format(len(batches)))
        sys.stderr.flush()
        for i, batch in enumerate(batches):
            # feed_dict
            feed_dict = self.make_feed_dict(batch)
            if _feed_dict: feed_dict.update(_feed_dict)

            run_results.extend(self.session.run(op, feed_dict=feed_dict))
            sys.stderr.write('{:5.3f} sec/step, steps: {:4}/{:4}\t\r'.format(
                (time.time() - start_time)/(i + 1), i+1, len(batches)))
            sys.stderr.flush()

        return [sum((x.tolist() for x in items), []) for items in zip(*run_results)]


    def make_session(self, sess=None, checkpoint=None, reuse_session=True):
        if sess is not None:
            assert sess.graph is self.graph
            self.session = sess
        else:
            if reuse_session and self.session:
                return

            if (not reuse_session) and self.session: self.session.close()

            checkpoint = checkpoint or self.checkpoint
            assert checkpoint is not None

            with self.graph.as_default():
                session_config = tf.ConfigProto()
                session_config.allow_soft_placement = True
                session = tf.Session(config=session_config, graph=self.graph)
                self.session = session
                saver = tf.train.Saver()
                saver.restore(session, checkpoint)
                self.session = session

    
    def make_batches(self, x, y, batch_capacity=None):
        batch_capacity = batch_capacity or (self.batch_capacity * 5)
        return dataprocessing.make_batches_source_target_const_capacity_batch_from_list(
            x, y, self.params["vocab"]["source_dict"], self.params["vocab"]["target_dict"],
            self.params["vocab"]["UNK_ID"],
            self.params["vocab"]["EOS_ID"],
            self.params["vocab"]["PAD_ID"],
            batch_capacity,
            allow_skip=False)

    def calculate_sentence_perplexity(self, sources, targets):
        batches = self.make_batches(sources, targets)
        perp, = self.execute_op(self.op_perplexity, batches)
        return perp 


    def calculate_corpus_perplexity(self, sources, targets):
        batches = self.make_batches(sources, targets)
        sent_perp, = self.execute_op(self.op_perplexity, batches)
        sent_lens = np.array(sum((batch[1][1] for batch in batches), []))
        perp = np.exp(np.sum(np.log(sent_perp) * sent_lens) / np.sum(sent_lens))

        return perp


    def calculate_translation_score(self, sources, targets, length_penalty_a=None):
        if length_penalty_a is None:
            length_penalty_a = self.params["test"]["length_penalty_a"]
        batches = self.make_batches(sources, targets)
        scores = self.execute_op(self.op_trans_score, batches, {self.ph_length_penalty: length_penalty_a})
        return scores


    def translate_sentences(self, texts, beam_size=1, return_search_results=False, init_y_texts=None):

        # Translate
        batch_capacity = 5 * self.batch_capacity // beam_size
        if init_y_texts is None:
            init_y_texts = [''] * len(texts)

        batches = self.make_batches(texts, init_y_texts, batch_capacity)
        candidates, scores = self.execute_op(self.op_beam_hypos_scores, batches, {self.ph_beam_size: beam_size})


        if return_search_results:
            nsamples = len(candidates)
            # flatten 
            candidates = sum(candidates, []) # [nsamples*beam_size, length(variable)]
            # convert to string
            candidates = self.IDs2text(candidates) #[nsamples*beam_size]
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores
        else:
            # take top 1
            candidates = [beam[0] for beam in candidates] # [nsamples, length(variable)]
            # convert to string
            candidates = self.IDs2text(candidates) #[nsamples]

            return candidates

def main():
    # logger
    basicConfig(level=INFO)

    # mode keys
    TRANSLATE = 'translate'
    PERPLEXITY = 'perplexity'
    CORPUS_PERP = 'corpus_perplexity'
    TRANS_DETAIL = 'translate_detail'
    modes = [TRANSLATE, PERPLEXITY, CORPUS_PERP, TRANS_DETAIL]

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('optional', type=str, nargs='*')
    parser.add_argument('--model_dir', '--dir', '-d', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=modes, default=TRANSLATE)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpu_cores', type=int, default=None)
    parser.add_argument('--sampling_method', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_capacity', type=int, default=None)
    parser.add_argument('--context_delimiter', type=str, default=None)
    parser.add_argument('--beam_size', type=int, default=1)
    args = parser.parse_args()

    inference = Inference(
        args.model_dir,
        checkpoint = args.checkpoint,
        n_gpus = args.n_gpus,
        n_cpu_cores = args.n_cpu_cores,
        batch_capacity = args.batch_capacity,
        sampling_method = args.sampling_method)

    inference.make_session()

    if args.mode == TRANSLATE or args.mode == TRANS_DETAIL:
        if args.context_delimiter is not None:
            x, y = zip(*(line.split(args.context_delimiter) for line in sys.stdin))
        else:
            x, y = [line.strip() for line in sys.stdin], None

        if args.mode == TRANSLATE:
            for line in inference.translate_sentences(x, args.beam_size, init_y_texts=y):
                print(line)
        else:
            hyp, score = inference.translate_sentences(x, args.beam_size, True, init_y_texts=y)
            for _hyp, _score in zip(hyp, score):
                for sent,sent_score in zip(_hyp, _score): 
                    print('{}\t{}'.format(sent_score, sent))
                print('')


    elif args.mode == PERPLEXITY or args.mode == CORPUS_PERP:
        src_f, trg_f = args.optional[0], args.optional[1]
        with open(src_f, 'r') as f:
            x = [line.strip() for line in f]
        with open(trg_f, 'r') as f:
            y = [line.strip() for line in f]

        if args.mode == PERPLEXITY:
            print(*inference.calculate_sentence_perplexity(x, y))
        else:
            print(inference.calculate_corpus_perplexity(x, y))


if __name__ == '__main__':
    main()
