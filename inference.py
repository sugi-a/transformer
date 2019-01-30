import tensorflow as tf
import numpy as np
import argparse
import sys
import os
import codecs
import subprocess
import time

from nltk.translate.bleu_score import corpus_bleu

from graph import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    #insert model_config's dir prior to this script's dir'''
    sys.path.insert(0, args.model_dir)

import model_config
print("model_config has been loaded from {}".format(model_config.__file__))
from model_config import Config as conf

def make_dataset_from_tok_file(tok_data_file, vocab_file_name, graph=None):
    """make a dataset input to the encoder
    
    Args:
        tok_data_file: shape (), dtype: tf.string. string must be stripped
        vocab_file_name: name of vocabulary file
        graph: graph in which the token-to-ID lookup table is placed
        
    Returns:
        dataset:
            tuple of ID sequence and its length. EOS is added.
            shape: ([None], []), dtype: (tf.int32, tf.int32)"""

    if graph is None:
        graph = tf.get_default_graph()

    with graph.as_default():
        table = tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=conf.UNK_ID,
                key_column_index=0)
        return tf.data.TextLineDataset(tok_data_file)\
            .map(lambda line: tf.string_split([line]).values)\
            .map(lambda tokens: tf.cast(table.lookup(tokens), tf.int32))\
            .map(lambda seq: tf.concat( #adding EOS ID
                [seq, tf.ones([1], tf.int32)*model_config.Config.EOS_ID], axis=0))\
            .map(lambda seq: (seq, tf.shape(seq)[0]))

def make_dataset_from_texts(texts, vocab_file_name, graph=None):
    """make a dataset input to the encoder
    
    Args:
        texts: list of str. texts must not contain \n
        vocab_file_name: name of the vocabulary file
        graph: graph in which the dataset is made
    Returns:
        dataset: tuple of ID sequence and its length. EOS is added to the seq.
        shape: ([None], []), dtype: (tf.int32, tf.int32)"""

    if graph is None:
        graph = tf.get_default_graph()

    texts = [line.strip() for line in texts]
    seqs = conf.tokenize_source(texts)

    with graph.as_default():
        return tf.data.Dataset.from_generator(lambda:seqs,
                                              tf.int32, tf.TensorShape([None]))\
            .map(lambda seq: tf.concat( #adding EOS ID
                [seq, tf.ones([1], tf.int32)*model_config.Config.EOS_ID], axis=0))\
            .map(lambda seq: (seq, tf.shape(seq)[0]))

class Inference:
    def __init__(self, graph=None, checkpoint=None): 
        """Build translator
        
        Args:
            graph: the graph under in which the inference network is built.
                   if None, a new graph will be made.
            checkpoint: Name of the checkpoint from which the value of parameters will be restored. You can specify it later.
        """
        
        self.checkpoint = checkpoint
        if graph is None:
            graph = tf.Graph()
        self.graph = graph

        hp = model_config.Hyperparams

        with self.graph.as_default():
            # encoder. expand dimension of the inputs since the encoder recieves batched inputs.
            #dummy dataset
            #_dummy_dataset = tf.data.Dataset.from_tensors(
                #tf.placeholder(tf.int32, shape=[None])
            #).batch(1) #out_shapes = (?,?)
            self.inputs_itr = tf.data.Iterator.from_structure(
                (tf.int32, tf.int32),
                (tf.TensorShape((None, None)), tf.TensorShape([None]))
            )
            #self.inputs_itr = tf.data.Iterator.from_structure(_dummy_dataset.output_types,
                                                         #_dummy_dataset.output_shapes)
            x, lengths = self.inputs_itr.get_next()
            self.encoder = Encoder(x, lengths, hp, False)

            # ------------ decoder --------------
            #decoder inputs
            self.dec_inputs_ph = tf.placeholder(tf.int32, [None, None])
            #add <s> to the head
            _dec_inputs = tf.concat([
                tf.ones((tf.shape(self.dec_inputs_ph)[0],1), dtype=tf.int32) * conf.SOS_ID,
                self.dec_inputs_ph], axis=-1)

            #decoder lengths. all the sequences in a batch have the same length
            self.dec_lengths_ph = tf.placeholder(tf.int32, [])
            _dec_lengths = tf.tile(tf.expand_dims(self.dec_lengths_ph, axis=0),
                                   [tf.shape(self.dec_inputs_ph)[0]])
            #encoder hidden states
            #recieves numpy array of the cached encoder hidden states and mask
            #[batch_size(N), sentence_len, embed_size]
            self.enc_hidden_states_ph = tf.placeholder(tf.float32,
                                                       [None, None, hp.embed_size]) 
            self.enc_mask_ph = tf.placeholder(tf.bool, [None, None]) #[N, sentence_len]

            #tile the place holders to match the first dimension with the decoder inputs
            n_tiles = (tf.shape(self.dec_inputs_ph)[0] //
                       tf.shape(self.enc_hidden_states_ph)[0])
            #encoder hidden states [beam_batch_len, sentence_len, embed_size]
            _enc_hidden_states = tf.tile(self.enc_hidden_states_ph,
                [n_tiles, 1, 1]) 
            #encoder mask [beam_batch_len, sentence_len]
            _enc_mask = tf.tile(self.enc_mask_ph,
                [n_tiles, 1]) 

            #build Decoder
            self.decoder = Decoder(_dec_inputs,
                                   _dec_lengths,
                                   _enc_hidden_states,
                                   _enc_mask,
                                   hp,
                                   False)

            #additional operation nodes for prediction
            self.last_pos_logits = self.decoder.logits[:, -1]
            self.last_pos_softmax = self.decoder.softmax_outputs[:, -1]
            


    def get_1_best_for_dataset(self, dataset, feed_dict=None, checkpoint=None):
        """translate each sentence in dataset by greedy search.
        
        Args:
            dataset: shape: ([None], ())
            feed_dict: feed_dict given when evaluating the encoder
        Returns:
            candidates:
                a list of numpy array.
                candidates[j] is the candidate for the j-th source sentence
        """
        with self.graph.as_default():
            if feed_dict is None:
                feed_dict = {}
            
            #checkpoint
            if checkpoint is None:
                checkpoint = self.checkpoint
                assert(checkpoint is not None)

            #batch the dataset
            dataset = dataset.padded_batch(
                model_config.Hyperparams.batch_size,
                ([None], []),
                (conf.PAD_ID, 0)
            ).prefetch(1)
            
            #graph nodes for greedy search
            top1_IDs = tf.argmax(self.last_pos_logits, axis=1, output_type=tf.int32) #[N]

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            #the list to be returned
            ret_cands = []

            with tf.Session(config=config) as sess:
                #initialize tables
                sess.run(tf.tables_initializer())
                #restore network parameters
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)

                #initialize input iterator
                sess.run(self.inputs_itr.make_initializer(dataset), feed_dict=feed_dict)

                #run encoder
                encoder_outputs_batches = []
                encoder_masks_batches = []
                while True:
                    try:
                        enc_hidden_states_batch, enc_mask_batch = sess.run(
                            [self.encoder.outputs, self.encoder.enc_mask],
                            feed_dict=feed_dict
                        )
                        encoder_outputs_batches.append(enc_hidden_states_batch)
                        encoder_masks_batches.append(enc_mask_batch)
                    except:
                        break

                n_batches = len(encoder_outputs_batches)
                print("#source sentences batches:{}".format(n_batches))

                step = 0
                start_time = time.time()
                for enc_hidden_states, enc_mask in zip(encoder_outputs_batches,
                                                       encoder_masks_batches):
                    #log
                    time_elapsed = time.time() - start_time
                    time_remaining = 100000 if step==0 else (
                                     time_elapsed / step * (n_batches - step))
                    sys.stdout.write("translating {}-th batch. finished in {} sec\r"
                        .format(step, time_remaining))
                    step = step + 1
                    
                    #initialize beam candidates
                    batch_size = len(enc_hidden_states) #batch size (N)
                    beam_cands = np.zeros([batch_size, 0], np.int) #[N, 0]
                    beam_has_EOS = np.zeros([batch_size], bool) #[N]
                    #position to be predicted
                    position = 0

                    #iterate search
                    while True:
                        #calc probability distribution
                        top1_IDs_output = sess.run(top1_IDs, 
                                 feed_dict={self.enc_hidden_states_ph: enc_hidden_states,
                                            self.enc_mask_ph: enc_mask,
                                            self.dec_inputs_ph: beam_cands,
                                            self.dec_lengths_ph: position+1}) #[N, pos+1, Vocab]

                        #update candidate seqs and scores in the batch
                        beam_cands = np.concatenate([
                            beam_cands,
                            np.expand_dims(top1_IDs_output, axis=1) #[N, 1]
                        ], axis=1) #[N, pos+1]

                        beam_has_EOS = np.logical_or(beam_has_EOS,
                                                     np.equal(top1_IDs_output, conf.EOS_ID))

                        #whether to break
                        if np.all(beam_has_EOS):
                            break
                        elif position > len(enc_hidden_states[0]) * 2:
                            beam_cands[:, position] = conf.EOS_ID
                            beam_has_EOS[np.logical_not(beam_has_EOS)] = True
                            break

                        position = position + 1

                    ret_cands.extend(list(beam_cands))
            print("translation done.")
            return ret_cands

    def get_n_best_for_dataset(self, dataset, n, feed_dict=None, checkpoint=None):
        """translate each sentence in dataset into n candidates.
        
        Args:
            dataset: shape: ([None], ())
            n: beam width
            feed_dict: feed_dict given when evaluating the encoder
        Returns:
            a tuple (candidates, scores)
            candidates:
                a list of list of numpy array.
                candidates[j][i] is the i-th candidate for the j-th source sentence
            scores:
                list of list of float scores
                scores correspond to the candidates
        """
        with self.graph.as_default():
            if feed_dict is None:
                feed_dict = {}
            
            #checkpoint
            if checkpoint is None:
                checkpoint = self.checkpoint
                assert(checkpoint is not None)

            #batch the dataset
            dataset = dataset.padded_batch(
                model_config.Hyperparams.batch_size,
                ([None], []),
                (conf.PAD_ID, 0)
            ).prefetch(1)

            #additional op nodes
            #scores:[N, n], IDs:[N, n]
            top_n_scores_op, top_n_IDs_op = tf.math.top_k(self.last_pos_softmax, n, False)

            #the list to be returned
            ret_cands = []
            ret_scores = []

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                #initialize tables
                sess.run(tf.tables_initializer())
                #restore network parameters
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)

                #initialize input iterator
                sess.run(self.inputs_itr.make_initializer(dataset), feed_dict=feed_dict)

                #run encoder
                encoder_outputs = []
                encoder_masks = []
                while True:
                    try:
                        enc_hidden_states_batch, enc_mask_batch = sess.run(
                            [self.encoder.hidden_states, self.encoder.enc_mask],
                            feed_dict=feed_dict
                        )
                        encoder_outputs.extend(enc_hidden_states_batch)
                        encoder_masks.extend(enc_mask_batch)
                    except:
                        break

                n_samples = len(encoder_outputs)
                print("#source sentences:{}".format(n_samples))

                step = 0
                start_time = time.time()
                for _enc_hidden_states, _enc_mask in zip(encoder_outputs, encoder_masks):
                    #making a batch with 1 sample
                    enc_hidden_states = np.expand_dims(_enc_hidden_states, 0)
                    enc_mask = np.expand_dims(_enc_mask, 0)

                    #log
                    time_elapsed = time.time() - start_time
                    step_remaining = n_samples - step
                    time_remaining = 100000 if step==0 else time_elapsed / step * step_remaining
                    sys.stdout.write("translating {}-th sent. finished in {} sec\r".format(
                        step, time_remaining))
                    step = step + 1
                    
                    #initialize beam candidates
                    beam_cands = np.array([np.array([], np.int)])
                    beam_scores = np.array([0])
                    beam_has_EOS = np.array([False])
                    #position to be predicted
                    position = 0

                    #iterate search
                    while True:
                        #make batch of candidates without EOS
                        batch = np.stack(beam_cands[np.logical_not(beam_has_EOS)], axis=0) #[c, pos]
                        #calc probability distribution
#                        softmax_outputs = sess.run(self.decoder.softmax_outputs, 
                        top_n_IDs, top_n_scores = sess.run([top_n_IDs_op, top_n_scores_op], 
                                 feed_dict={self.enc_hidden_states_ph: enc_hidden_states,
                                            self.enc_mask_ph: enc_mask,
                                            self.dec_inputs_ph: batch,
                                            self.dec_lengths_ph: position+1}) #[c, pos+1, Vocab]

                        #update candidate seqs and scores in the batch
                        extendeds = np.concatenate([
                            np.repeat(batch, n, axis=0), #[c*n, pos]
                            np.reshape(top_n_IDs, [-1, 1]) #[c*n, 1]
                        ], axis=1) #[c*n, pos+1]
                        extended_scores =\
                            np.repeat(beam_scores[np.logical_not(beam_has_EOS)], n)\
                            + np.log(np.reshape(top_n_scores, [-1]) + np.finfo(np.float).eps) #[c*n]
                        extended_has_EOS = np.equal(extendeds[:, position], conf.EOS_ID)

                        #update beam candidates
                        beam_cands = np.array(
                            list(beam_cands[beam_has_EOS]) + list(extendeds))
                        beam_scores = np.array(
                            list(beam_scores[beam_has_EOS]) + list(extended_scores))
                        beam_has_EOS = np.array(
                            list(beam_has_EOS[beam_has_EOS]) + list(extended_has_EOS))

                        #take n best
                        length_penalty = np.power(
                            (5 + np.array([len(seq) for seq in beam_cands]))/ (5 + 1),
                            model_config.Hyperparams.length_penalty_a)
                        score_with_penalty = beam_scores / length_penalty
                        nbest_args = np.argsort(score_with_penalty, axis=0)[::-1][:n]
                        beam_cands = beam_cands[nbest_args]
                        beam_scores = beam_scores[nbest_args]
                        beam_has_EOS = beam_has_EOS[nbest_args]

                        #whether to break
                        if np.all(beam_has_EOS):
                            break
                        elif position > len(enc_hidden_states[0]) * 2:
                            for seq in beam_cands[np.logical_not(beam_has_EOS)]:
                                seq[position] = conf.EOS_ID
                            beam_scores[np.logical_not(beam_has_EOS)] = -np.inf
                            beam_has_EOS[np.logical_not(beam_has_EOS)] = True
                            break

                        position = position + 1

                    ret_cands.append(list(beam_cands))
                    ret_scores.append(list(beam_scores))
            print("translation done.")
            return (ret_cands, ret_scores)

    def translate_sentences(self, sentences, beam_width=1, checkpoint=None):
        """translate sentences
        
        Args:
            sentences: list of str. white spaces at the end of sentences will be
                removed before translation.
        
        Returns:
            translations. list of str"""

        #make sure sentences don't have \n at the ends
        sentences = [sent.strip() for sent in sentences]

        dataset = make_dataset_from_texts(sentences, conf.vocab_source, self.graph)

        if beam_width == 1:
            translations = self.get_1_best_for_dataset(dataset, checkpoint=checkpoint)
        else:
            translations, _ = self.get_n_best_for_dataset(dataset, beam_width, checkpoint=checkpoint)
            translations = [cands[0] for cands in translations]

        #detokenize
        translations = [seq.tolist() for seq in translations]
        translations = conf.detokenize_target(translations)

        return translations

    def translate_file(self, file_name, beam_width=1, checkpoint=None):
        """translate sentences in a file.
        sentence must not be tokenized. This method makes Dataset AFTER tokenizing the sentences (tokenization is not included in the Dataset pipeline)
        
        Returns:
            list of str"""
        with codecs.open(file_name, "r", "utf-8") as f:
           return self.translate_sentences(f.readlines(), beam_width, checkpoint)

    def BLEU_evaluation_with_test_data(self, beam_width=1, checkpoint=None,
                                       source_tok_file=None, target_raw_file=None):
        """Perform evaluation with BLEU metric using the test data"""

        #source and target files. source file must be tokenized
        assert (source_tok_file is None) == (target_raw_file is None)
        if source_tok_file is None:
            source_tok_file = conf.source_test_tok
        if target_raw_file is None:
            target_raw_file = conf.target_test

        #make dataset of source sentence
        test_source = make_dataset_from_tok_file(source_tok_file,
                                                 conf.vocab_source,
                                                 self.graph)
        
        #translate
        if beam_width == 1:
            translation = self.get_1_best_for_dataset(test_source, checkpoint=checkpoint)
        else:
            translation, _scores = self.get_n_best_for_dataset(test_source,
                                                               beam_width,
                                                               checkpoint=checkpoint)
            translation = [cands[0] for cands in translation]
        
        #detokenize
        translation = [seq.tolist() for seq in translation]
        translations = conf.detokenize_target(translation)

        #load reference
        with codecs.open(target_raw_file, 'r', 'utf-8') as _ref_file:
            references = [line.rstrip() for line in _ref_file]

        #tokenize source and reference for BLEU
        translations_tok = conf.bleu_tokenize_target(translations)
        references_tok = conf.bleu_tokenize_target(references)

        #calc BLEU
        references_tok = [[sent] for sent in references_tok]
        score = corpus_bleu(references_tok, translations_tok)

        return score

