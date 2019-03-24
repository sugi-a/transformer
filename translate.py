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
logger_handler = StreamHandler()
logger_handler.setLevel(DEBUG)
logger.addHandler(logger_handler)

from nltk.translate.bleu_score import corpus_bleu

from model import *
from utils import *
import dataprocessing


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
        
        if model is None:
            if graph is None:
                graph = tf.get_default_graph()
            self.graph = graph

            self.model = Transformer(Hyperparams, Config)
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
                (tf.int32, tf.int32),
                (tf.TensorShape((None, None)), tf.TensorShape([None]))
            )

            self.inputs_parallel = [self.inputs_itr.get_next() for i in range(n_gpus)]
            
            # computation graph
            self.beam_size_ph = tf.placeholder(tf.int32, [])
            def _beam_search(inputs):
                x, x_len = inputs
                beam_candidates, scores = model.decode(x, x_len, beam_size_ph, return_search_results=True)
                return beam_candidates, scores
            self.beam_candidates, self.scores = compute_parallel_and_concat(_beam_search, self.inputs_parallel, concat_device='/cpu:0')

             
    def do_beam_search(self, dataset, beam_size, session=None, checkpoint=None):
        """conduct beam search producing results as numpy arrays.
        Args:
            session: `tf.Session` to be used. If `None`, a new one is created. If specified, its `graph`
                must be the same as self.graph and all the variables used must be initialized before calling
                this method.
        Returns:
            A tuple of two numpy arrays: beam candidates and their scores.
            The structure is ([batch_size, beam_size, length], [batch_size, beam_size])"""
        
        session_to_close = None
        with self.graph.as_default():
            if session is None:
                checkpoint = checkpoint or self.checkpoint
                assert checkpoint is not None

                session_config = tf.ConfigProto()
                session_config.allow_soft_placement = True
                session = tf.Session(config=session_config)
                session_to_close = session

                # initialization
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

                # restoration of variables
                saver = tf.train.Saver()
                saver.restore(session, checkpoint)
            else:
                assert session.graph is self.graph

            # batch dataset
            dataset = dataset.padded_batch(Hyperparams.batch_size // self.n_gpus // beam_size,
                                            ([None], []),
                                            (Config.PAD_ID, 0))
            dataset = dataset.prefetch(self.n_gpus * 2)

            # initialization of the input Iterator
            session.run(self.inputs_itr.make_initializer(dataset))

            # beam search
            run_results = []
            while True:
                try:
                    run_results.append(session.run((self.beam_candidates, self.scores),
                                                   feed_dict={self.beam_size_ph: beam_size}))
                except tf.errors.OutOfRangeError:
                    break
            candidates, scores = [np.concatenate(x, axis=0) for x in zip(*run_results)]

            session_to_close.close()
        
        return candidates, scores


    def translate_sentences(self, texts, beam_size=1, return_search_results=False, checkpoint=None, session=None):
        """translate texts
        Args:
            texts: list of str. texts must be tokenized into subwords
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
        candidates, scores = self.do_beam_search(dataset, beam_size, session, checkpoint)

        if return_search_results:
            batch_size = len(candidates)
            # flatten and convert to list
            candidates = candidates.reshape([batch_size * beam_size, -1]).tolist() #[batch*beam, length]
            candidates = Config.IDs2text(candidates, 'target')
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores.tolist()
        else:
            # take top 1
            candidates = candidates[:, 0] # [batch_size, length]
            # convert to string
            candidates = candidates.tolist()

            return candidates
        
    def translate_preprocessed_texts(self, texts, *args, **kwargs):
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
        self.translate_sentences(texts, *args, **kwargs)

    def BLEU_evaluation(self, source_file, target_file, beam_size=1, session=None, checkpoint=None, result_file_prefix=None):
        """
        Args:
            source_file, target_file: must be the subword format
        Returns:
            BLEU score. Translation result is written into `result_file_prefix`.translations and the score is
            written into `result_file_prefix`.score if result_file_prefix is not None
            """
        # read source and target files
        with codecs.open(source_file, 'r') as s_f, codecs.open(target_file, 'r') as t_f:
            source_texts = [line in s_f]
            target_texts = [line in t_f] 
        
        translations = self.translate_sentences(source_texts, beam_size, return_search_results=False, checkpoint=checkpoint, session=session)

        # detokenize targets and translations (which is, 'preprocessed format')
        detok = Config.tokens2text([line.strip().split() for line in (target_texts + translations)], 'target')

        # tokenize targets and translations to compute BLEU
        b_tok = Config.text2tokens_BLEU(detok)

        # make translations and references for computing BLEU
        bleu_translations = detok[:len(detok)/2]
        bleu_references = [[x] for x in detok[len(detok)/2:]]
        
        # compute BLEU
        score = corpus_bleu(bleu_references, bleu_translations)

        # write results into files
        if result_file_prefix is not None:
            results_file_name, score_file_name = result_file_prefix + '.results', result_file_prefix + '.score'
            with codecs.open(results_file_name, 'w') as r_f, codecs.open(score_file_name, 'w') as s_f:
                # source \n reference \n translation \n\n
                r_f.writelines(['{}\n{}\n{}\n\n'.format(s, r[0], t)
                    for s,r,t in zip(source_texts, bleu_references, bleu_translations)])
                s_f.write(str(score))

        return score

