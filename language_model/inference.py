import argparse, sys, os, time
import tensorflow as tf
import numpy as np
from logging import getLogger, INFO, DEBUG, basicConfig
logger = getLogger(__name__)

from .lm import DecoderLanguageModel
from ..components.utils import *
from ..components import dataprocessing
from .datasetloader import make_const_capacity_batch_list

class Inference:
    def __init__(self, model_dir, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=4, batch_capacity=None):

        # Model's working directory
        self.model_dir = model_dir

        # log dir
        self.logdir = self.model_dir + '/log'

        # Load configuration
        self.config = {'model_dir': model_dir}
        with open(self.model_dir + '/lm_config.py', 'r') as f:
            exec(f.read(), self.config)
        
        self.params = self.config["params"]
        params = self.params

        # Vocabulary utility
        self.vocab = dataprocessing.Vocabulary(
            params["vocab"]["dict"],
            UNK_ID= params["vocab"]["UNK_ID"],
            EOS_ID= params["vocab"]["EOS_ID"],
            PAD_ID= params["vocab"]["PAD_ID"])

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
                self.model = DecoderLanguageModel(params)
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
                'x_len': tf.placeholder(tf.int32, [None])
            }

            # Splitting for data parallel computing
            self.inputs_parallel = non_even_split(
                (self.ph_dict['x'], self.ph_dict['x_len']), self.n_gpus)
            
            # Computation graph for perplexity
            self.op_perplexity = self.make_op(self.fn_perplexity)


    def fn_perplexity(self, inputs):
        (x, x_len) = inputs
        logits = self.model.get_logits(x, training=False)
        is_target = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, x[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]
        perp = tf.exp(-seq_log_prob / tf.cast(x_len, tf.float32))

        return [perp]


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
        return {self.ph_dict['x']: batch[0],
                    self.ph_dict['x_len']: batch[1]}


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

    
    def make_batches(self, x, batch_capacity=None):
        batch_capacity = batch_capacity or self.batch_capacity

        data = self.vocab.text2IDs(x, False)
        batches = make_const_capacity_batch_list(data, [len(d) for d in data], batch_capacity, self.vocab.PAD_ID)
        return batches


    def calculate_sentence_perplexity(self, x):
        batches = self.make_batches(x)
        perp, = self.execute_op(self.op_perplexity, batches)
        return perp 


    def calculate_corpus_perplexity(self, x):
        batches = self.make_batches(x)
        sent_perp, = self.execute_op(self.op_perplexity, batches)
        sent_lens = np.array(sum((batch[1] for batch in batches), []))
        perp = np.exp(np.sum(np.log(sent_perp) * sent_lens) / np.sum(sent_lens))

        return perp



def main():
    # logger
    basicConfig(level=INFO)

    # mode keys
    PERPLEXITY = 'perplexity'
    CORPUS_PERP = 'corpus_perplexity'
    modes = [PERPLEXITY, CORPUS_PERP]

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '--dir', '-d', type=str, default='.')
    parser.add_argument('--mode', type=str, choices=modes, default=PERPLEXITY)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpu_cores', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_capacity', type=int, default=None)
    args = parser.parse_args()

    inference = Inference(
        args.model_dir,
        checkpoint = args.checkpoint,
        n_gpus = args.n_gpus,
        n_cpu_cores = args.n_cpu_cores,
        batch_capacity = args.batch_capacity)

    inference.make_session()

    if args.mode == PERPLEXITY or args.mode == CORPUS_PERP:
        x = [line.strip() for line in sys.stdin]

        if args.mode == PERPLEXITY:
            for p in inference.calculate_sentence_perplexity(x):
                print(p)
        else:
            print(inference.calculate_corpus_perplexity(x))


if __name__ == '__main__':
    main()
