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
    def __init__(self, model=None, graph=None, checkpoint=None, n_gpus=1, n_cpu_cores=8): 
        """Build translator
        
        Args:
            graph: the graph under in which the inference network is built.
                   if None, the default graph will be used.
            checkpoint: Name of the checkpoint from which the value of parameters will be restored. You can specify it later.
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
                # parallel inputs must be taken from the Iterator in the right order
                self.inputs_parallel = non_even_split(self.inputs_itr.get_next(), self.n_gpus)
            
            # computation graph
            self.beam_size_ph = tf.placeholder(tf.int32, [])
            def _beam_search(inputs):
                (x, x_len), (init_y, init_y_len) = inputs
                beam_candidates, scores = self.model.decode(
                    x,
                    x_len,
                    self.beam_size_ph,
                    return_search_results=True,
                    init_y=init_y,
                    init_y_len=init_y_len)
                return beam_candidates, scores
            self.beam_candidates_scores = compute_parallel(_beam_search, self.inputs_parallel) # [n_gpus]

             
    def do_beam_search(self, dataset, beam_size, session=None, checkpoint=None, reuse_session=True, init_y_dataset=None):
        """conduct beam search producing results as numpy arrays.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created and reused.
                If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
        Returns:
            A tuple of two lists: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length(variable)], [batch_size, beam_size])"""
        
        logger.info('Beam search decoding.')
        session_to_close = None
        with self.graph.as_default():
            if session is None:
                if reuse_session and self.session:
                    logger.info('Reusing session')
                    session = self.session
                else:
                    logger.info('Create new Session to perform beam search.')
                    checkpoint = checkpoint or self.checkpoint
                    assert checkpoint is not None

                    session_config = tf.ConfigProto()
                    session_config.allow_soft_placement = True
                    session = tf.Session(config=session_config)

                    if reuse_session:
                        try:
                            self.session.close()
                        except:
                            pass
                        self.session = session
                    else:
                        session_to_close = session

                    # initialization
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

                    # restoration of variables
                    saver = tf.train.Saver()
                    saver.restore(session, checkpoint)
            else:
                assert session.graph is self.graph

            # dataset of the initial translations given
            if init_y_dataset is None:
                # Create init_y_dataset (empty sequences) if not specified
                init_y_dataset = tf.data.Dataset.from_tensors((tf.zeros([0], dtype=tf.int32), 0))
                init_y_dataset = init_y_dataset.repeat()

            # merge the main inputs and the initial translation datasets
            dataset = tf.data.Dataset.zip((dataset, init_y_dataset))

            # batch dataset
            #batch_size = (Hyperparams.batch_size // self.n_gpus) * 4 // (beam_size ** 2)
            batch_size = (Hyperparams.batch_size) * 2 // (beam_size) + 1
            dataset = dataset.padded_batch(batch_size,
                                            (([None], []), ([None], [])),
                                            ((Config.PAD_ID, 0), (Config.PAD_ID, 0)))
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
                sys.stderr.write('{} sec/sample, samples processed: {}   \r'.format(
                    (time.time() - start_time)/iter_count/batch_size,
                    iter_count * batch_size))

            # candidates: [batch_size, beam_size, length(variable)], scores: [batch_size, beam_size]
            candidates, scores = [sum([array.tolist() for array in arrays], []) for arrays in zip(*run_results)]

            if session_to_close is not None:
                session_to_close.close()
        
        return candidates, scores


    def translate_sentences(self, texts, beam_size=1, return_search_results=False, checkpoint=None, session=None, reuse_session=True, init_y_texts=None):
        """translate texts. Input format should be tokenized (subword) one and the output's is preprocessed.
        Args:
            texts: list of str. texts must be tokenized into subwords before the call of this method
        Returns:
            If return_search_results is True, returns list of candidates (list of list of str) and scores
            (list of list of float). Each text is in the target language in preprocessed format (not tokenized). 
            If return_search_results is False, returns the translations with MAP (list of str).
            """

        dataset = dataprocessing.make_dataset_single_from_texts(texts,
                                                                Config.vocab_source,
                                                                Config.UNK_ID,
                                                                Config.EOS_ID,
                                                                self.n_cpu_cores)
        if init_y_texts is not None:
            assert len(texts) == len(init_y_texts)
            init_y_dataset = dataprocessing.make_dataset_single_from_texts(init_y_texts,
                                                                Config.vocab_target,
                                                                Config.UNK_ID,
                                                                Config.EOS_ID,
                                                                self.n_cpu_cores)
        else:
            init_y_dataset = None

        candidates, scores = self.do_beam_search(dataset, beam_size, session, checkpoint, reuse_session, init_y_dataset)

        if return_search_results:
            nsamples = len(candidates)
            # flatten 
            candidates = sum(candidates, []) # [nsamples*beam_size, length(variable)]
            # convert to string
            candidates = Config.IDs2text(candidates, 'target') #[nsamples*beam_size]
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores
        else:
            # take top 1
            candidates = [beam[0] for beam in candidates] # [nsamples, length(variable)]
            # convert to string
            candidates = Config.IDs2text(candidates, 'target') #[nsamples]

            return candidates
        
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
            results_file_name, score_file_name = result_file_prefix + '.results', result_file_prefix + '.score'
            with codecs.open(results_file_name, 'w') as r_f, codecs.open(score_file_name, 'w') as s_f:
                # source \n reference \n translation \n\n
                r_f.writelines(['{}\n{}\n{}\n\n'.format(s, ' '.join(r[0]), ' '.join(t))
                    for s,r,t in zip(source_texts, bleu_references, bleu_translations)])
                s_f.write(str(score))

        if return_samples:
            return score, source_texts, bleu_translations, bleu_references
        else:
            return score

