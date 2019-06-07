import tensorflow as tf
import numpy as np
import argparse
import sys
import os
import codecs
import subprocess
import time
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger('Translator')
logger.setLevel(DEBUG)

from nltk.translate.bleu_score import corpus_bleu

from model import *
from utils import *
import dataprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
#insert model_config's dir prior to this script's dir'''
    if sys.path[0] != args.model_dir:
        sys.path.insert(0, args.model_dir)

import model_config
logger.info("model_config has been loaded from {}".format(model_config.__file__))
from model_config import Config
from model_config import Hyperparams

class Inference:
    # Data input method keys
    DATASET = 'dataset'
    PLACE_HOLDER = 'placeholder'

    def __init__(self, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=8, input_method=PLACE_HOLDER, batch_capacity=None, sampling_method=None):
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

        self.session = None
        
        if model is None:
            if graph is None:
                graph = tf.get_default_graph()
            self.graph = graph

            with self.graph.as_default():
                self.model = Transformer(Hyperparams, Config)
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
            # Make inputs. self.inputs_parallel is built and,
            # if method=dataset, self.inputs_itr is built
            # if method=placeholder, self.ph_dict is built
            assert input_method == Inference.DATASET or input_method == Inference.PLACE_HOLDER
            self.input_method = input_method

            if batch_capacity is None:
                batch_capacity = Hyperparams.batch_size * Hyperparams.maxlen
            self.batch_capacity = batch_capacity

            if self.input_method == Inference.DATASET:
                # input Iterator
                self.inputs_itr = tf.data.Iterator.from_structure(
                    ((tf.int32, tf.int32), (tf.int32, tf.int32)),
                    ((tf.TensorShape([None, None]), tf.TensorShape([None])),
                    (tf.TensorShape([None, None]), tf.TensorShape([None])))
                )

                if self.n_gpus == 1:
                    # Single GPU
                    self.inputs_parallel = [self.inputs_itr.get_next()]
                else:
                    # parallel inputs must be taken from the Iterator in the same order as the original
                    # so `get_next()` should be called only once.
                    self.inputs_parallel = non_even_split(self.inputs_itr.get_next(), self.n_gpus)
            else:
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
            self.sequence_log_prob = tf.placeholder(tf.bool, [])
            def _perplexity(inputs):
                (x, x_len), (y, y_len) = inputs
                logits = self.model.get_logits(x, y, x_len, y_len, False)
                is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)

                logprobs = tf.math.log_softmax(logits, axis=-1) # [batch, length, vocab]
                seq_logprobs = tf.batch_gather(logprobs, y[:, :, None]) # [batch, length, 1]
                seq_logprobs = seq_logprobs[:, :, 0]
                log_perp = tf.reduce_sum(
                    tf.cond(self.sequence_log_prob,
                        lambda: seq_logprobs * is_target,
                        lambda: seq_logprobs * is_target / tf.reduce_sum(is_target, axis=1, keepdims=True)),
                    axis=1) # [batch]
                perp = tf.exp(log_perp) # [batch]
                return perp
            self.perplexity = compute_parallel(_perplexity, self.inputs_parallel)

    def prepare_session(self, checkpoint=None, reuse_session=True):
        '''
            '''
        if self.session and reuse_session:
            logger.info('Reusing session')
            session = self.session
        else:
            if self.session:
                self.session.close()

            checkpoint = checkpoint or self.checkpoint
            assert checkpoint is not None

            session_config = tf.ConfigProto()
            session_config.allow_soft_placement = True
            logger.info('Create new Session to perform beam search.')
            session = tf.Session(config=session_config, graph=self.graph)
            self.session = session
            saver = tf.train.Saver()
            saver.restore(session, checkpoint)
        return session

    def do_beam_search(self, dataset, beam_size, session=None, checkpoint=None, reuse_session=True):
        """conduct beam search producing results as numpy arrays.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created and reused.
                If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.input_method == Inference.DATASET
        
        logger.info('Beam search decoding.')
        with self.graph.as_default():
            if session is None:
                session = self.prepare_session(checkpoint=checkpoint, reuse_session=reuse_session)
            else:
                assert session.graph is self.graph

            dataset = dataset.prefetch(self.n_gpus * 2)

            # initialization of the input Iterator
            session.run(self.inputs_itr.make_initializer(dataset))

            # beam search
            run_results = []
            start_time = time.time()
            iter_count = 0
            while True:
                try:
                    run_results.extend(session.run(self.beam_candidates_scores,
                                                   feed_dict={self.beam_size_ph: beam_size}))
                except tf.errors.OutOfRangeError:
                    break
                iter_count += 1
                sys.stderr.write('{} sec/step. {}-th step.   \r'.format(
                    (time.time() - start_time)/iter_count, iter_count))

            # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
            candidates, scores = [sum([array.tolist() for array in arrays], []) for arrays in zip(*run_results)]

        return candidates, scores

    def do_beam_search_placeholder(self, batches, beam_size, session=None, checkpoint=None, reuse_session=True):
        """conduct beam search producing results as list of integers.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created and reused.
                If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
            batches: [batches: (([batch_size: [length: int]], [batch_size: int]),
                                ([batch_size: [length: int]], [batch_size: int]))]
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.input_method == Inference.PLACE_HOLDER

        logger.info('Beam search decoding.')
        with self.graph.as_default():
            if session is None:
                session = self.prepare_session(checkpoint=checkpoint, reuse_session=reuse_session)
            else:
                assert session.graph is self.graph

            # Beam search
            run_results = []
            start_time = time.time()
            iter_count = 0
            for batch in batches:
                run_results.extend(session.run(self.beam_candidates_scores,
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

    def do_calc_perplexity(self, dataset, sequence_log_prob=False, session=None, checkpoint=None, reuse_session=True):
        """conduct beam search producing results as numpy arrays.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created and reused.
                If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.input_method == Inference.DATASET
        
        logger.debug('Perplexity Calculation')
        with self.graph.as_default():
            if session is None:
                session = self.prepare_session(checkpoint=checkpoint, reuse_session=reuse_session)
            else:
                assert session.graph is self.graph

            dataset = dataset.prefetch(self.n_gpus * 2)

            # initialization of the input Iterator
            session.run(self.inputs_itr.make_initializer(dataset))

            # Calculation
            run_results = []
            start_time = time.time()
            iter_count = 0
            while True:
                try:
                    run_results.extend(session.run(self.perplexity, feed_dict={self.sequence_log_prob: sequence_log_prob}))
                except tf.errors.OutOfRangeError:
                    break
                iter_count += 1
                sys.stderr.write('{} sec/step. {}-th step.   \r'.format(
                    (time.time() - start_time)/iter_count, iter_count))

            # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
            perp = sum(map(lambda x:x.tolist(), run_results), [])
            return perp

    def do_calc_perplexity_placeholder(self, batches, sequence_log_prob=False, session=None, checkpoint=None, reuse_session=True):
        """conduct beam search producing results as list of integers.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created and reused.
                If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
            batches: [batches: (([batch_size: [length: int]], [batch_size: int]),
                                ([batch_size: [length: int]], [batch_size: int]))]
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""

        assert self.input_method == Inference.PLACE_HOLDER

        logger.info('Beam search decoding.')
        with self.graph.as_default():
            if session is None:
                session = self.prepare_session(checkpoint=checkpoint, reuse_session=reuse_session)
            else:
                assert session.graph is self.graph

            # Calculation of perplexity
            run_results = []
            start_time = time.time()
            iter_count = 0
            for batch in batches:
                run_results.extend(session.run(
                    self.perplexity,
                    feed_dict={
                        self.ph_dict['x']: batch[0][0],
                        self.ph_dict['x_len']: batch[0][1],
                        self.ph_dict['init_y']: batch[1][0],
                        self.ph_dict['init_y_len']: batch[1][1],
                        self.sequence_log_prob: sequence_log_prob}
                ))
                iter_count += 1
                sys.stderr.write('{} sec/step, steps: {}/{}   \r'.format(
                    (time.time() - start_time)/iter_count, iter_count, len(batches)))

            # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
            perp = sum(map(lambda x:x.tolist(), run_results), [])
            return perp

    def calculate_perplexity(self, sources, targets, sequence_log_prob=False, checkpoint=None, session=None, reuse_session=True):
        """Calculate the perplexity of the target text given a source text.
        Args:
            sources: [#samples: str]"""

        batch_capacity = self.batch_capacity

        if self.input_method == Inference.DATASET:
            dataset = dataprocessing.make_dataset_source_target_const_capacity_batch_from_list(
                sources, target,
                Config.vocab_source, Config.vocab_target, Config.UNK_ID, Config.EOS_ID,
                Config.PAD_ID, batch_capacity,
                batch_capacity=batch_capacity,
                sort=False,
                allow_skip=False)

            perp = self.do_calc_perplexity(dataset, sequence_log_prob=sequence_log_prob, session=session, checkpoint=checkpoint, reuse_session=reuse_session)
        else:
            batches = dataprocessing.make_batches_source_target_const_capacity_batch_from_list(
                sources, targets,
                Config.vocab_source, Config.vocab_target, Config.UNK_ID, Config.EOS_ID,
                Config.PAD_ID, batch_capacity,
                batch_capacity=batch_capacity,
                sort=False,
                allow_skip=False)

            perp = self.do_calc_perplexity_placeholder(batches, sequence_log_prob=sequence_log_prob, session=session, checkpoint=checkpoint, reuse_session=reuse_session)
        return perp 

    def translate_sentences(self, texts, beam_size=1, return_search_results=False, checkpoint=None, session=None, reuse_session=True, init_y_texts=None, return_in_subwords=False):
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

        if self.input_method == Inference.DATASET:

            dataset = dataprocessing.make_dataset_source_target_const_capacity_batch_from_list(texts,
                init_y_texts, Config.vocab_source, Config.vocab_target, Config.UNK_ID, Config.EOS_ID,
                Config.PAD_ID, batch_capacity,
                batch_capacity=batch_capacity,
                sort=False,
                allow_skip=False)

            candidates, scores = self.do_beam_search(dataset, beam_size, session, checkpoint, reuse_session)
        else:
            batches = dataprocessing.make_batches_source_target_const_capacity_batch_from_list(texts,
                init_y_texts, Config.vocab_source, Config.vocab_target, Config.UNK_ID, Config.EOS_ID,
                Config.PAD_ID, batch_capacity,
                batch_capacity=batch_capacity,
                sort=False,
                allow_skip=False)

            candidates, scores = self.do_beam_search_placeholder(batches, beam_size, session, checkpoint,
                reuse_session)


        if return_search_results:
            nsamples = len(candidates)
            # flatten 
            candidates = sum(candidates, []) # [nsamples*beam_size, length(variable)]
            # convert to string
            if return_in_subwords:
                candidates = Config.IDs2tokens(candidates, Config.TARGET)
            else:
                candidates = Config.IDs2text(candidates, Config.TARGET) #[nsamples*beam_size]
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores
        else:
            # take top 1
            candidates = [beam[0] for beam in candidates] # [nsamples, length(variable)]
            # convert to string
            if return_in_subwords:
                candidates = Config.IDs2tokens(candidates, Config.TARGET)
            else:
                candidates = Config.IDs2text(candidates, Config.TARGET) #[nsamples]

            return candidates

    def calc_perp_preprocessed_texts(self, sources, targets, *args, **kwargs):
        sources = [' '.join(toks) for toks in Config.text2tokens(sources, Config.SOURCE)]
        targets = [' '.join(toks) for toks in Config.text2tokens(targets, Config.TARGET)]
        return self.calculate_perplexity(sources, targets, *args, **kwargs)
        
    def calc_perp_raw_texts(self, sources, targets, *args, **kwargs):
        sources = Config.preprocess(sources, Config.SOURCE)
        targets = Config.preprocess(targets, Config.TARGET)
        return self.calc_perp_preprocessed_texts(sources, targets, *args, **kwargs)
        
    def translate_preprocessed_texts(self, texts, *args, init_y_texts=None, **kwargs):
        """translate texts
        Args:
            texts: list of str. texts must be in the preprocessed format (not tokenized by sentencepiece)
        Returns:
            If return_search_results is True, returns list of candidates (list of list of str) and scores
            (list of list of float). Each text is in the target language in preprocessed format (not tokenized). 
            If return_search_results is False, returns the translations with MAP (list of str).
            """
        # convert to the subword format
        texts = Config.text2tokens(texts, 'source') 
        texts = [' '.join(toks) for toks in texts]

        # context
        if init_y_texts is not None:
            init_y_texts = Config.text2tokens(init_y_texts, 'target') 
            init_y_texts = [' '.join(toks) for toks in init_y_texts]

        return self.translate_sentences(texts, *args, init_y_texts=init_y_texts, **kwargs)

    def translate_raw_texts(self, texts, *args, init_y_texts=None, **kwargs):
        # preprocess
        texts = Config.preprocess(texts, 'source')

        if init_y_texts is not None:
            init_y_texts = Config.preprocess(init_y_texts, 'target')

        return self.translate_preprocessed_texts(texts, *args, init_y_texts=init_y_texts, **kwargs)

    def BLEU_evaluation(self, source_file, target_file, beam_size=1, session=None, checkpoint=None, result_file_prefix=None, return_samples=False):
        """
        Args:
            source_file, target_file: must be the subword format
        Returns:
            BLEU score. Translation result is written into `result_file_prefix`.translations and the score is
            written into `result_file_prefix`.score if result_file_prefix is not None
            """
        # read source and target files
        with codecs.open(source_file, 'r') as s_f, codecs.open(target_file, 'r') as t_f:
            source_texts = [line.strip() for line in s_f]
            target_texts = [line.strip() for line in t_f] 
            assert len(source_texts) == len(target_texts)       
        
        translations = self.translate_sentences(source_texts, beam_size, return_search_results=False, checkpoint=checkpoint, session=session)

        # detokenize targets (which is, 'preprocessed format')
        detok_reference = Config.tokens2text([line.split() for line in target_texts], 'target')

        # tokenize targets and translations to compute BLEU (concatenate the two for faster computation)
        b_tok = Config.text2tokens_BLEU(translations + detok_reference)

        # make translations and references for computing BLEU
        bleu_translations = b_tok[:len(b_tok)//2]
        bleu_references = [[x] for x in b_tok[len(b_tok)//2:]]
        
        # compute BLEU
        try:
            # Custom metric if defined
            score = Config.compute_bleu(bleu_references, bleu_translations)
            logger.info('Evaluated with the custom metric.')
        except:
            # nltk's corpus_bleu()
            score = corpus_bleu(bleu_references, bleu_translations)
            logger.info('Evaluated with the default corpus_bleu of nltk')

        # write results into files
        if result_file_prefix is not None:
            write_BLEU_results_to_file(result_file_prefix, source_texts, score, bleu_references, bleu_translations)

        if return_samples:
            return score, source_texts, bleu_translations, bleu_references
        else:
            return score

def write_BLEU_results_to_file(prefix, sources, score, references, translations):
    """
    references and translations are those you give to nltk's corpus_bleu()
    assert len(sources) == len(references) == len(translations)
        """
    results_file_name, score_file_name = prefix + '.results', prefix + '.score'
    source_f_name, ref_f_name, out_f_name= [prefix + '.' + ext for ext in ['src', 'ref', 'out']]
    with codecs.open(results_file_name, 'w') as r_f,\
        codecs.open(score_file_name, 'w') as s_f,\
        codecs.open(source_f_name, 'w') as src_f,\
        codecs.open(ref_f_name, 'w') as ref_f,\
        codecs.open(out_f_name, 'w') as out_f:
        # source \n reference \n translation \n\n
        r_f.writelines(['{}\n{}\n{}\n\n'.format(s, ' '.join(r[0]), ' '.join(t))
            for s,r,t in zip(sources, references, translations)])
        s_f.write(str(score) + '\n')
        src_f.writelines(['{}\n'.format(s) for s in sources])
        ref_f.writelines(['{}\n'.format(' '.join(r[0])) for r in references])
        out_f.writelines(['{}\n'.format(' '.join(t)) for t in translations])
