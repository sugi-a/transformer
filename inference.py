import argparse, sys, os, codecs, subprocess, time
from collections import deque
import tensorflow as tf
import numpy as np
from logging import getLogger, INFO, basicConfig
logger = getLogger('Translator')

from components.model import *
from components.utils import *
from components import dataprocessing

class Inference:
    def __init__(self, model_config, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=8, batch_capacity=8192, sampling_method=None):
        """Build translator
        
        Args:
            graph: the graph under in which the inference network is built.
                   if None, the default graph will be used.
            checkpoint: Name of the checkpoint from which the value of parameters will be restored. You can specify it later.
            sampling_method: if None or 0, the normal beam search is applied.
                if 1, sampling method [Edunov+, 2018] is used, in which beam_size is ignored.
        """
        self.n_cpu_cores = n_cpu_cores
        self.n_gpus = n_gpus
        params = model_config.params
        self.params = params

        self.session = None

        # decode ids into subwords
        if hasattr(model_config, 'IDs2tokens'):
            self.IDs2tokens = model_config.IDs2tokens
            logger.debug('Inference loaded a custom IDs2tokens')

        self.PAD_ID = params["vocab"]["PAD_ID"]
        self.EOS_ID = params["vocab"]["EOS_ID"]
        self.SOS_ID = params["vocab"]["SOS_ID"]
        self.UNK_ID = params["vocab"]["UNK_ID"]
        with codecs.open(params["vocab"]["target_dict"]) as f:
            word_list = [line.split()[0] for line in f]
        self.target_dict = {id: word for id, word in enumerate(word_list)}
        self.target_dict[params["vocab"]["UNK_ID"]] = '#'
        logger.debug('Inference uses the default IDs2tokens')
        
        if model is None:
            if graph is None:
                graph = tf.get_default_graph()
            self.graph = graph

            with self.graph.as_default():
                self.model = Transformer(params)
                if self.n_gpus == 1:
                    # Single GPU
                    with tf.device(None):
                        self.model.instanciate_vars()
                else:
                    # place variables in the device /cpu:0
                    with tf.device('/cpu:0'):
                        self.model.instanciate_vars()
        else:
            self.graph = model.graph
            self.model = model

        self.checkpoint = checkpoint

        with self.graph.as_default():
            self.batch_capacity = batch_capacity

            self.ph_dict = {
                'x': tf.placeholder(tf.int32, [None, None]),
                'x_len': tf.placeholder(tf.int32, [None]),
                'init_y': tf.placeholder(tf.int32, [None, None]),
                'init_y_len': tf.placeholder(tf.int32, [None])
            }
            self.inputs_parallel = non_even_split(
                ((self.ph_dict['x'], self.ph_dict['x_len']),
                (self.ph_dict['init_y'], self.ph_dict['init_y_len'])), self.n_gpus)
            
            # computation graph for beam search
            self.beam_size_ph = tf.placeholder(tf.int32, [])
            def _beam_search(inputs):
                (x, x_len), (init_y, init_y_len) = inputs
                beam_candidates, scores = self.model.decode(
                    x,
                    x_len,
                    self.beam_size_ph,
                    return_search_results=True,
                    init_y=init_y,
                    init_y_len=init_y_len,
                    sampling_method=sampling_method)
                return beam_candidates, scores
            self.beam_candidates_scores = compute_parallel(_beam_search, self.inputs_parallel) # [n_gpus]

            # Computation graph for calculation of perplexity
            self.trans_score = tf.placeholder(tf.bool, [])
            def _perplexity(inputs):
                (x, x_len), (y, y_len) = inputs
                logits = self.model.get_logits(x, y, x_len, y_len, False)
                is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)

                log_prob_dist = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
                log_prob = tf.batch_gather(log_prob_dist, y[:, :, None]) # [batch, length, 1]
                log_prob = log_prob[:, :, 0]
                seq_log_prob = tf.reduce_sum(log_prob * is_target, axis=1) #[batch]
                ret = tf.cond(self.trans_score,
                    lambda: seq_log_prob / tf.cast(tf.pow((5 + y_len)/(1 + 5),
                        self.params["test"]["length_penalty_a"]), tf.float32),  # trans. score
                    lambda: tf.exp(seq_log_prob / tf.cast(y_len, tf.float32)) # perplexity
                )
                return ret
            self.perplexity = compute_parallel(_perplexity, self.inputs_parallel)

    def IDs2tokens(self, IDs):
        return [ ' '.join(self.target_dict[id] for id in sent
            if id != self.PAD_ID and id != self.EOS_ID and id != self.SOS_ID) for sent in IDs ]

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

    def do_beam_search(self, batches, beam_size):
        """conduct beam search producing results as list of integers.
        Args:
            batches: [batches: (([batch_size: [length: int]], [batch_size: int]),
                                ([batch_size: [length: int]], [batch_size: int]))]
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.session
        logger.debug('Beam search decoding.')

        # Beam search
        run_results = []
        start_time = time.time()
        iter_count = 0
        for batch in batches:
            run_results.extend(self.session.run(self.beam_candidates_scores,
                feed_dict={
                    self.beam_size_ph: beam_size,
                    self.ph_dict['x']: batch[0][0],
                    self.ph_dict['x_len']: batch[0][1],
                    self.ph_dict['init_y']: batch[1][0],
                    self.ph_dict['init_y_len']: batch[1][1]}))
            iter_count += 1
            sys.stderr.write('{} sec/step, steps: {}/{}   \r'.format(
                (time.time() - start_time)/iter_count, iter_count, len(batches)))

        # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
        candidates, scores = [sum([array.tolist() for array in arrays], []) for arrays in zip(*run_results)]

        return candidates, scores

    def do_calc_perplexity(self, batches, trans_score=False):
        """conduct beam search producing results as list of integers.
        Args:
            batches: [batches: (([batch_size: [length: int]], [batch_size: int]),
                                ([batch_size: [length: int]], [batch_size: int]))]
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.session

        # Calculation of perplexity
        run_results = []
        start_time = time.time()
        iter_count = 0
        for batch in batches:
            run_results.extend(self.session.run(
                self.perplexity,
                feed_dict={
                    self.ph_dict['x']: batch[0][0],
                    self.ph_dict['x_len']: batch[0][1],
                    self.ph_dict['init_y']: batch[1][0],
                    self.ph_dict['init_y_len']: batch[1][1],
                    self.trans_score: trans_score}
            ))
            iter_count += 1
            sys.stderr.write('{} sec/step, steps: {}/{}   \r'.format(
                (time.time() - start_time)/iter_count, iter_count, len(batches)))

        # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
        perp = sum(map(lambda x:x.tolist(), run_results), [])
        return perp

    def calculate_perplexity(self, sources, targets, trans_score=False):
        """Calculate the perplexity of the target text given a source text.
        Args:
            sources: [#samples: str]"""

        batch_capacity = self.batch_capacity

        batches = dataprocessing.make_batches_source_target_const_capacity_batch_from_list(
            sources, targets,
            self.params["vocab"]["source_dict"],
            self.params["vocab"]["target_dict"],
            self.UNK_ID,
            self.EOS_ID,
            self.PAD_ID,
            batch_capacity,
            allow_skip=False)

        perp = self.do_calc_perplexity(batches, trans_score=trans_score)
        return perp 

    def translate_sentences(self, texts, beam_size=1, return_search_results=False, init_y_texts=None):
        """translate texts. Input format should be tokenized (subword) one and the output's is preprocessed.
        Args:
            texts: list of str. texts must be tokenized into subwords before the call of this method
            init_y_texts: prefix of the translations. The output translations includes them.
        Returns:
            If return_search_results is True, returns list of candidates and scores
            (list of list of float).
            List of candidates is a array (nested list) whose shape is:
                If `return_search_results` is True:
                    [batch_size, beam_size]
                If `return_search_results` is False (default):
                    [batch_size] which contains the translations with MAP 
            and each element is 
                if `return_in_subwords` is True:
                    list of subword tokens
                if `return_in_subwords` is False (default):
                    str which have been produced by decoding the subword sequence
            """

        # Translate
        batch_capacity = self.batch_capacity // beam_size
        logger.debug('batch capacity: {}'.format(batch_capacity))

        if init_y_texts is None:
            init_y_texts = [''] * len(texts)

        batches = dataprocessing.make_batches_source_target_const_capacity_batch_from_list(texts,
            init_y_texts,
            self.params["vocab"]["source_dict"],
            self.params["vocab"]["target_dict"],
            self.UNK_ID,
            self.EOS_ID,
            self.PAD_ID,
            batch_capacity,
            allow_skip=False)

        candidates, scores = self.do_beam_search(batches, beam_size)


        if return_search_results:
            nsamples = len(candidates)
            # flatten 
            candidates = sum(candidates, []) # [nsamples*beam_size, length(variable)]
            # convert to string
            candidates = self.IDs2tokens(candidates) #[nsamples*beam_size]
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores
        else:
            # take top 1
            candidates = [beam[0] for beam in candidates] # [nsamples, length(variable)]
            # convert to string
            candidates = self.IDs2tokens(candidates) #[nsamples]

            return candidates

def main():
    # logger
    basicConfig(level=INFO)

    # keys
    TRANSLATE = 'translate'
    PERPLEXITY = 'perplexity'

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default=TRANSLATE)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_cpu_cores', type=int, default=None)
    parser.add_argument('--sampling_method', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_capacity', type=int, default=None)
    parser.add_argument('--context_delimiter', type=str, default=None)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--basedir', type=str, default=None)
    args = parser.parse_args()

    # model's working directory, where the config file is placed
    model_dir = os.path.abspath(args.model_dir)

    # load model_config.py
    sys.path.insert(0, model_dir)
    import model_config
    params = model_config.params

    # log directory. (model dir)/log
    logdir = model_dir + '/log'

    # Change directory if specified in the command line params or config.
    os.chdir(args.basedir or params["basedir"] or '.')
    
    
    # load checkpoint from MODELDIR/log/sup_checkpoint/ (if not specified)
    checkpoint = args.checkpoint or tf.train.latest_checkpoint(logdir + '/sup_checkpoint')
    assert checkpoint is not None; logger.debug('checkpoint', checkpoint)

    # batch_capacity (specified or decided according to n_gpus)
    batch_capacity = args.batch_capacity or 64*128*args.n_gpus

    inference = Inference(
        model_config, checkpoint=checkpoint, n_gpus=args.n_gpus,
        batch_capacity=args.batch_capacity, sampling_method=args.sampling_method)
    inference.make_session()

    if args.mode == TRANSLATE:
        if args.context_delimiter is not None:
            x, y = zip(*(line.split(args.context_delimiter) for line in sys.stdin))
            for line in inference.translate_sentences(x, args.beam_size, init_y_texts=y):
                print(line)
        else:
            x = [line.strip() for line in sys.stdin]
            for line in inference.translate_sentences(x, args.beam_size):
                print(line)


if __name__ == '__main__':
    main()
