import argparse, sys, os, time, json
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np
from logging import getLogger, INFO, DEBUG, basicConfig
logger = getLogger(__name__)

from .model import *
from .utils import compute_parallel, merge_nested_dict, non_even_split
from . import dataprocessing as dp
from .decoding import BeamSearchKeys, length_penalty


class InferenceOpPH:
    """Inference operator container with tf.placeholder"""
    def __init__(self, op_fn, data_ph):
        self.flat_data_ph = nest.flatten(data_ph)
        self.op_fn = op_fn
        self.__op = None


    @property
    def op(self):
        if self.__op is None:
            self.__op = self.op_fn()
        return self.__op


    def make_feed_dict(self, batch):
        flatten_batch = nest.flatten(dp.list2numpy_nested(batch))
        return {ph: bat for ph, bat in zip(self.flat_data_ph, flatten_batch)}


class InferenceOpDS:
    """Inference operator container with tf.data.Dataset"""
    def __init__(self, op_fn, input_iter):
        self.op_fn = op_fn
        self.input_iter = input_iter

    
    @property
    def op(self):
        if self.__op is None:
            self.__op = self.op_fn()
        return self.__op


    def make_dataset(self, batch_generator, nprefetch=2):
        return tf.data.Dataset.from_generator(
            batch_generator,
            self.input_iter.output_types,
            self.input_iter.output_shapes
        ).prefetch(nprefetch)


class OperationWithPlaceholderFeed:
    def __init__(self, op_fn, data_phs, param_phs=None, graph=None):
        self.data_phs = data_phs
        self.param_phs = param_phs or {}
        self.op_fn = op_fn
        self.__op = None
        self.graph = graph or tf.get_default_graph()


    @property
    def op(self):
        if self.__op is None:
            with self.graph.as_default():
                self.__op = self.op_fn()
        return self.__op


    def make_feed_dict(self, batch, param_feeds=None):
        flatten_batch = nest.flatten_up_to(self.data_phs, batch)
        flatten_batch_phs = nest.flatten(self.data_phs)
        ret = {ph: bat for ph, bat in zip(flatten_batch_phs, flatten_batch)}

        param_feeds = param_feeds or {}
        ret.update({self.param_phs[k]: v for k,v in param_feeds.items()})
        return ret


class Inference:
    def __init__(self, model_dir, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=4, batch_capacity=None, decode_config=None):

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

        # Merging the default decoding setting and the one specified at runtime.
        self.decoding = params['inference']['decoding']
        merge_nested_dict(self.decoding, decode_config)
        logger.debug('Decode configuration: ' + str(self.decoding))

        # Vocabulary utility
        self.vocab = dp.Vocabulary(
            params["vocab"]["target_dict"],
            UNK_ID = params["vocab"]["UNK_ID"],
            EOS_ID = params["vocab"]["EOS_ID"],
            PAD_ID = params["vocab"]["PAD_ID"],
            SOS_ID = params["vocab"]["SOS_ID"])
        
        self.src_vocab = dp.Vocabulary(
            params['vocab']['source_dict'],
            UNK_ID = params['vocab']['UNK_ID'],
            EOS_ID = params['vocab']['EOS_ID'],
            PAD_ID = params['vocab']['PAD_ID'],
            SOS_ID = params["vocab"]["SOS_ID"])

        # Select the method to convert IDs to tokens
        if 'IDs2text' in self.config:
            self.IDs2text = self.config["IDs2text"]
            logger.debug('Inference using the custom IDs2text method')
        else:
            self.IDs2text = self.vocab.IDs2text
            logger.debug('Inference using the default IDs2text method')
        
        # Computing options
        self.n_cpu_cores = n_cpu_cores
        self.n_gpus = n_gpus or 1
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
            # Default placeholder for input data (paired sentences)
            self.default_phs = (
                (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                )


    def fn_beam_search(self, inputs, beam_size):
        (x, x_len), (init_y, init_y_len) = inputs
        beam_candidates, scores = self.model.decode_V2(
            x,
            x_len,
            init_y,
            init_y_len,
            beam_size = beam_size,
            return_search_results = True,
            decode_config = self.decoding)
        return beam_candidates, scores

    def fn_perplexity(self, inputs):
        (x, x_len), (y, y_len) = inputs
        y_in, y_out = y[:, :-1], y[:, 1:]
        _y_len = y_len - 1

        logits = self.model.get_logits(x, y_in, x_len, _y_len, False)
        is_target = tf.sequence_mask(_y_len, tf.shape(y_in)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, y_out[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]
        perp = tf.exp(-seq_log_prob / tf.cast(_y_len, tf.float32))

        return [perp]


    def fn_translation_score(self, inputs, length_penalty_a):
        (x, x_len), (y, y_len) = inputs
        y_in, y_out = y[:, :-1], y[:, 1:]
        _y_len = y_len - 1

        logits = self.model.get_logits(x, y_in, x_len, _y_len, False)
        is_target = tf.sequence_mask(_y_len, tf.shape(y_in)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, y_out[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]

        # penalty coefficient (defined in model.py)
        penalty = length_penalty(_y_len, length_penalty_a)

        score = seq_log_prob / penalty 
        
        return [score]

    def make_op(self, fn, data_phs, *args, param_phs=None, **kwargs):
        """Create operation which computes the function specified with GPU(s) in parallel.
        Args:
            fn:
                Args: inputs = ((x, x_len), (y, y_len))
                Returns: tuple or list (ret1, ret2, ...). ret: [BATCH_SIZE, ...]
        Returns:
            list of replicated operations to be computed in parallel.
            """

        with self.graph.as_default():
            parallel_inputs = non_even_split(data_phs, self.n_gpus)
            op_fn = lambda: compute_parallel(fn, parallel_inputs, *args, **kwargs)
            return OperationWithPlaceholderFeed(op_fn, data_phs, param_phs=param_phs)


    def execute_op_iter(self, op, batches_iter, **feeds):
        assert self.session

        for batch in batches_iter:
            feed_dict = op.make_feed_dict(batch, feed)
            
            res = self.session.run(op.op, feed_dict=feed_dict)
            for bat_res in res:
                for items in zip(*bat_res):
                    yield items

        
    def execute_op(self, op, batches, **feeds):
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
            feed_dict = op.make_feed_dict(batch, feeds)

            run_results.extend(self.session.run(op.op, feed_dict=feed_dict))
            sys.stderr.write('{:5.3f} sec/step, steps: {:4}/{:4}\t\r'.format(
                (time.time() - start_time)/(i + 1), i+1, len(batches)))
            sys.stderr.flush()

        return [sum((x.tolist() for x in items), []) for items in zip(*run_results)]


    def make_session(self, sess=None, load_checkpoint=None):
        assert self.session is None
        if sess is not None:
            assert sess.graph is self.graph
            self.session = sess
            if load_checkpoint:
                saver = tf.train.Saver(tf.global_variables(self.model.scope_name))
                saver.restore(self.session, self.checkpoint)
        else:
            with self.graph.as_default():
                session_config = tf.ConfigProto()
                session_config.allow_soft_placement = True
                self.session = tf.Session(config=session_config, graph=self.graph)
                saver = tf.train.Saver(tf.global_variables(self.model.scope_name))
                saver.restore(self.session, self.checkpoint)

    
    def make_batches(self, *args, **kwargs):
        return list(self.make_batches_iter(*args, **kwargs))


    def make_batches_iter(self, x, y, batch_capacity=None):
        batch_capacity = batch_capacity or (self.batch_capacity * 5)
        x_IDs = dp.gen_line2IDs(x, self.src_vocab)
        y_IDs = dp.gen_line2IDs(y, self.vocab)
        return dp.gen_dual_const_capacity_batch(zip(x_IDs, y_IDs), batch_capacity, self.vocab.PAD_ID)


    def calculate_sentence_perplexity(self, sources, targets):
        if not hasattr(self, 'op_perplexity'):
            self.op_perplexity = self.make_op(self.fn_perplexity, self.default_phs)

        batches = self.make_batches(sources, targets)
        perp, = self.execute_op(self.op_perplexity, batches)
        return perp 


    def calculate_corpus_perplexity(self, sources, targets):
        if not hasattr(self, 'op_perplexity'):
            self.op_perplexity = self.make_op(self.fn_perplexity, self.default_phs)

        batches = self.make_batches(sources, targets)
        sent_perp, = self.execute_op(self.op_perplexity, batches)
        sent_lens = np.array(sum((batch[1][1] for batch in batches), []))
        perp = np.exp(np.sum(np.log(sent_perp) * sent_lens) / np.sum(sent_lens))

        return perp


    def calculate_translation_score(self, sources, targets, length_penalty_a=None):
        if not hasattr(self, 'op_trans_score'):
            # Computation graph for translation score
            with self.graph.as_default():
                params = {'length_penalty_a': tf.placeholder(tf.float64, [])}
                self.op_trans_score = self.make_op(
                    self.fn_translation_score,
                    self.default_phs,
                    **params,
                    param_phs = params)

        if length_penalty_a is None:
            length_penalty_a = self.decoding['length_penalty_a']
        batches = self.make_batches(sources, targets)
        scores, = self.execute_op(self.op_trans_score, batches, length_penalty_a=length_penalty_a)
        return scores


    def translate_sentences(self, texts, beam_size=1, return_search_results=False, init_y_texts=None):

        if not hasattr(self, 'op_beam_hypos_scores'):
            # computation graph for beam search
            with self.graph.as_default():
                param = {'beam_size': tf.placeholder(tf.int32, [])}
                self.op_beam_hypos_scores = self.make_op(
                    self.fn_beam_search,
                    self.default_phs,
                    **param,
                    param_phs = param)
            
        # Translate
        batch_capacity = 5 * self.batch_capacity // beam_size
        if init_y_texts is None:
            header = self.params['inference']['header']
            if type(header) == int:
                header = self.vocab.ID2tok[header]
            elif header is None:
                header = ''
            init_y_texts = [header] * len(texts)

        batches = self.make_batches(texts, init_y_texts, batch_capacity)
        candidates, scores = self.execute_op(self.op_beam_hypos_scores, batches, beam_size=beam_size)

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
    # mode keys
    TRANSLATE = 'translate'
    PERPLEXITY = 'perplexity'
    CORPUS_PERP = 'corpus_perplexity'
    TRANS_DETAIL = 'translate_detail'
    TRANS_SCORE = 'trans_score'
    modes = [TRANSLATE, PERPLEXITY, CORPUS_PERP, TRANS_DETAIL, TRANS_SCORE]

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '--dir', '-d', type=str, default='.')
    parser.add_argument('optional', type=str, nargs='*')
    parser.add_argument('--mode', type=str, choices=modes, default=TRANSLATE)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpu_cores', type=int, default=None)
    parser.add_argument('--decode-config-json', '--config', '-c', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_capacity', type=int, default=None)
    parser.add_argument('--context_delimiter', type=str, default=None)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--log-level', choices=["INFO", "DEBUG"], default="INFO")
    args = parser.parse_args()

    # logger
    basicConfig(level=(INFO if args.log_level == 'INFO' else 'DEBUG'))

    dec_conf = json.loads(args.decode_config_json) if args.decode_config_json is not None else None

    inference = Inference(
        args.model_dir,
        checkpoint = args.checkpoint,
        n_gpus = args.n_gpus,
        n_cpu_cores = args.n_cpu_cores,
        batch_capacity = args.batch_capacity,
        decode_config = dec_conf)

    inference.make_session()

    if args.mode == TRANSLATE or args.mode == TRANS_DETAIL:
        def __trans(x ,y):
            if args.mode == TRANSLATE:
                for line in inference.translate_sentences(x, args.beam_size, init_y_texts=y):
                    print(line)
            else:
                hyp, score = inference.translate_sentences(x, args.beam_size, True, init_y_texts=y)
                for _hyp, _score in zip(hyp, score):
                    for sent,sent_score in zip(_hyp, _score): 
                        print('{}\t{}'.format(sent_score, sent))
            sys.stdout.flush()
        
        if args.online:
            while True:
                try:
                    line = sys.stdin.readline()
                    if len(line) == 0:
                        exit(0)
                    if args.context_delimiter is not None:
                        x, y = line.split(args.context_delimiter)
                        x, y = [x], [y]
                    else:
                        x, y = [line], None
                    __trans(x, y)
                except Exception as e:
                    sys.stderr.write(e)
        else:
            if args.context_delimiter is not None:
                x, y = zip(*(line.split(args.context_delimiter) for line in sys.stdin))
            else:
                x, y = [line.strip() for line in sys.stdin], None
            __trans(x, y)
         
    elif args.mode == PERPLEXITY or args.mode == CORPUS_PERP or args.mode == TRANS_SCORE:
        src_f, trg_f = args.optional[0], args.optional[1]
        with open(src_f, 'r') as f:
            x = [line.strip() for line in f]
        with open(trg_f, 'r') as f:
            y = [line.strip() for line in f]

        if args.mode == PERPLEXITY:
            for p in inference.calculate_sentence_perplexity(x, y):
                print(p)
        elif args.mode == TRANS_SCORE:
            for s in inference.calculate_translation_score(x, y):
                print(s)
        else:
            print(inference.calculate_corpus_perplexity(x, y))


if __name__ == '__main__':
    main()
