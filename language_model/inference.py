import argparse, sys, os, time
import tensorflow as tf
import numpy as np
from logging import getLogger, INFO, DEBUG, basicConfig
logger = getLogger(__name__)

from .language_model import DecoderLanguageModel, load_model_config
from ..components.utils import *
from ..components import dataprocessing as dp
from ..components.inference import Inference as MTInference

class Inference(MTInference):
    def __init__(self, model_dir, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=4, batch_capacity=None):

        # Model's working directory
        self.model_dir = model_dir

        # Load configuration
        self.config = load_model_config(model_dir, 'lm_config.py')
        self.params = params = self.config["params"]

        # Vocabulary utility
        self.vocab = dp.Vocabulary(
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
        self.checkpoint = checkpoint or tf.train.latest_checkpoint(os.path.join(model_dir, 'log', 'sup_checkpoint'))

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
            self.default_phs = (
                tf.placeholder(tf.int32, [None, None]),
                tf.placeholder(tf.int32, [None]))


    def fn_perplexity(self, inputs):
        (x, x_len) = inputs
        x_in, x_out = x[:, :-1], x[:, 1:]
        io_len = x_len - 1
        seq_log_prob, = self.fn_log_prob(inputs)
        perp = tf.exp(-seq_log_prob / tf.cast(io_len, tf.float32))

        return [perp]


    def fn_log_prob(self, inputs):
        (x, x_len) = inputs
        x_in, x_out = x[:, :-1], x[:, 1:]
        io_len = x_len - 1

        logits = self.model.get_logits(x_in, training=False)
        is_target = tf.sequence_mask(io_len, tf.shape(x_in)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1)
        log_prob = tf.batch_gather(log_prob_dist, x_out[:,:, None])[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1)

        return [seq_log_prob]


    def fn_cond_log_prob(self, inputs):
        (x, x_len), (c, c_len) = inputs
        mcl, mxl = tf.shape(c)[1], tf.shape(x)[1]
        ids = tf.range(mcl + mxl)[None] + (
            1 - tf.sequence_mask(c_len, mcl + mxl, dtype=tf.int32)) * (mcl - c_len)[:, None]
        ids = tf.minimum(ids, mcl + mxl - 1)
        joint = tf.batch_gather(tf.concat([c, x], axis=-1), ids)
        j_len = x_len + c_len
        j_in, j_out = joint[:, :-1], joint[:, 1:]

        logits = self.model.get_logits(j_in, training=False)
        is_target = tf.sequence_mask(j_len - 1, tf.shape(j_in)[1], dtype=tf.float32
            ) - tf.sequence_mask(c_len - 1, tf.shape(j_in)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1)
        log_prob = tf.batch_gather(log_prob_dist, j_out[:,:, None])[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1)
        return [seq_log_prob]


    def make_feed_dict(self, batch):
        return {self.ph_dict['x']: batch[0],
                    self.ph_dict['x_len']: batch[1]}


    def make_batches_iter(self, x, batch_capacity=None, header=True, footer=True):
        batch_capacity = batch_capacity or self.batch_capacity

        IDs = dp.gen_line2IDs(x, self.vocab)
        return dp.gen_const_capacity_batch(
            IDs,
            batch_capacity,
            self.vocab.PAD_ID)


    def make_multi_sentence_batch_gen(self, x, batch_capacity=None):
        """
        x: [(line1-1, line1-2, ..), (line2-1, line2-2, ...), ...]
            """
        batch_capacity = batch_capacity or self.batch_capacity
        IDs = (tuple(self.vocab.line2IDs(line) for line in row) for row in x)
        return dp.gen_const_capacity_batch_multi_seq(IDs, batch_capacity, self.vocab.PAD_ID)



    def calculate_sentence_perplexity(self, x):
        if not hasattr(self, 'op_perplexity'):
            self.op_perplexity = self.make_op(self.fn_perplexity, self.default_phs)

        batches = self.make_batches(x)
        perp, = self.execute_op(self.op_perplexity, batches)
        return perp 


    def calculate_corpus_perplexity(self, x):
        if not hasattr(self, 'op_perplexity'):
            self.op_perplexity = self.make_op(self.fn_perplexity, self.default_phs)

        batches = self.make_batches(x)
        sent_perp, = self.execute_op(self.op_perplexity, batches)
        sent_lens = np.array(sum((batch[1] for batch in batches), []))
        perp = np.exp(np.sum(np.log(sent_perp) * sent_lens) / np.sum(sent_lens))

        return perp


    def calculate_log_prob(self, x):
        if not hasattr(self, 'op_log_prob'):
            self.op_log_prob = self.make_op(self.fn_log_prob, self.default_phs)
        batches = self.make_batches(x)
        logprob, = self.execute_op(self.op_log_prob, batches)
        return logprob


    def calculate_cond_log_prob(self, c, x):
        if not hasattr(self, 'op_c_log_prob'):
            with self.graph.as_default():
                phs = (
                    (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                    (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                    )
                self.op_c_log_prob = self.make_op(self.fn_cond_log_prob, phs)
        batches = list(self.make_multi_sentence_batch_gen(zip(x, c)))
        logprob, = self.execute_op(self.op_c_log_prob, batches)
        return logprob


    def calculate_pmi(self, c, x, sep=None, head=None):
        """PMI(c, x; head, sep)
        = log p(x | head c sep) - log p(x | head sep)
        = log p(c sep x | head) - log p(c sep | head) - log p(sep x | head) + log p(sep | head)"""
        n = len(c)
        if sep is None:
            sep = [''] * n
        elif type(sep) == str:
            sep = [sep] * n
        if head is None:
            head = [''] * n
        elif type(head) == str:
            head = [head] * n

        assert n == len(x) == len(sep) == len(head)

        probs = self.calculate_cond_log_prob(
            ['{} {} {}'.format(_h, _c, _s) for _h, _c, _s in zip(head, c, sep)] +
            ['{} {}'.format(_h, _s) for _h, _s in zip(head, sep)],
            x + x)
        probs = np.array(probs)
        return probs[:n] - probs[n:n*2]


    def calculate_pmi_v2(self, c, null_c, x):
        n = len(c)
        if type(null_c) == str:
            null_c = [null_c] * n
        assert n == len(null_c) == len(x)

        probs = self.calculate_cond_log_prob(c + null_c, x + x)
        probs = np.array(probs)
        return probs[:n] - probs[n:]


def main():
    # logger
    basicConfig(level=INFO)

    # mode keys
    PERPLEXITY = 'ppl'
    CORPUS_PERP = 'corpus_ppl'
    LOG_PROB = 'log_prob'
    PMI = 'pmi'
    modes = [PERPLEXITY, CORPUS_PERP, LOG_PROB, PMI]

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

    if args.mode == PERPLEXITY or args.mode == CORPUS_PERP or args.mode == LOG_PROB:
        x = [line.strip() for line in sys.stdin]

        if args.mode == PERPLEXITY:
            for p in inference.calculate_sentence_perplexity(x):
                print(p)
        elif args.mode == CORPUS_PERP:
            print(inference.calculate_corpus_perplexity(x))
        elif args.mode == LOG_PROB:
            for p in inference.calculate_log_prob(x):
                print(p)
    elif args.mode == PMI:
        head, c, sep, x = zip(*map(lambda x: x.split('\t'), sys.stdin))
        for p in inference.calculate_pmi(c, x, sep, head):
            print(p)


if __name__ == '__main__':
    main()

