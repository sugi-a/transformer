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

            # Splitting for data parallel computing
            self.default_parallel_inputs = non_even_split(self.default_phs, self.n_gpus)
            
            # Computation graph for perplexity
            self.op_perplexity = self.make_op(self.fn_perplexity)


    def fn_perplexity(self, inputs):
        (x, x_len) = inputs
        x_in, x_out = x[:, :-1], x[:, 1:]
        io_len = x_len - 1

        logits = self.model.get_logits(x_in, training=False)
        is_target = tf.sequence_mask(io_len, tf.shape(x_in)[1], dtype=tf.float32)

        log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
        log_prob = tf.batch_gather(log_prob_dist, x_out[:, :, None]) # [batch, length, 1]
        log_prob = log_prob[:, :, 0]
        seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]
        perp = tf.exp(-seq_log_prob / tf.cast(io_len, tf.float32))

        return [perp]


    def make_feed_dict(self, batch):
        return {self.ph_dict['x']: batch[0],
                    self.ph_dict['x_len']: batch[1]}


    def make_batches_iter(self, x, batch_capacity=None, header=True, footer=True):
        batch_capacity = batch_capacity or self.batch_capacity

        IDs = dp.gen_line2IDs(x, self.vocab)
        _header = self.params['vocab']['sent_header']
        _footer = self.params['vocab']['sent_footer']
        if header and _header is not None:
            IDs = ([_header] + ids for ids in IDs)
        if footer and _footer is not None:
            IDs = (ids + [_footer] for ids in IDs)
        return dp.gen_const_capacity_batch(
            IDs
            batch_capacity,
            self.vocab.PAD_ID)


    def make_multi_sentence_batch_gen(self, x, batch_capacity=None):
        """
        x: [(line1-1, line1-2, ..), (line2-1, line2-2, ...), ...]
            """
        batch_capacity = batch_capacity or self.batch_capacity
        header = self.params['vocab']['sent_header']
        footer = self.params['vocab']['sent_footer']
        def __concat(x):
            ret = []
            for line in x:
                if header is not None:
                    ret.append(header)
                ret.extend(self.vocab.line2IDs(line))
                if footer is not None:
                    ret.append(footer)
            yield ret
        IDs = map(__concat, x)
        return dp.gen_const_capacity_batch(IDs, batch_capacity, self.vocab.PAD_ID)



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


    def calculate_log_prob(self, x):
        batches = self.make_batches(x)
        sent_perp, = self.execute_op(self.op_perplexity, batches)
        sent_lens = np.array(sum((batch[1] for batch in batches), []))
        logprob = - np.log(sent_perp) * sent_lens
        return logprob


    def calculate_pmi(self, context, x):
        """PMI(context, x) = log p(context^x) - log p(context) - log p(x)"""
        assert len(context) == len(x)
        n = len(context)
        probs = self.calculate_log_prob([c + ' ' + _x for c, _x in zip(context, x)] + context + x)
        return probs[:n] - probs[n:n * 2] - probs[n*2:]


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

    # for PMI
    parser.add_argument('--context', '-c', type=str, default=None)
    parser.add_argument('--sentence', '-x', type=str, default=None)
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
        with open(args.context) as f:
            context = f.readlines()
        with open(args.sentence) as f:
            sentence = f.readlines()
        
        for p in inference.calculate_pmi(context, sentence):
            print(p)


if __name__ == '__main__':
    main()

