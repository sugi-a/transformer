import sys, os, codecs
from logging import getLogger, StreamHandler, DEBUG

import tensorflow as tf
import numpy as np
import sentencepiece as spm


def text2tokens(sents, model_file):
    """tokenize sentences into sequences of tokens
    
    Args:
        sents: list of str
        type: "source" or "target"
        
    Returns:
        list of list of str"""
    
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return [sp.EncodeAsPieces(sent) for sent in sents]

def text2IDs(sents, model_file):
    """tokenize sentences into sequences of IDs
    
    Args:
        sents: list of str
        
    Returns:
        list of list of int"""
    #in this model_config.py target=ja
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return [sp.EncodeAsIds(sent) for sent in sents]

def tokens2text(tokens, model_file):
    """detokenize tokens into strings
    Args:
        tokens: list of list of str
        type: "source" or "target"
    
    Returns:
        list of str"""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_config)
    return [sp.DecodePieces(tok) for tok in tokens]

def IDs2text(seqs, model_file):
    """detokenize sequence of IDs into strings
    Args:
        source_seqs: list of list of int
    
    Returns:
        list of str"""
    #in this model_config.py source=en
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return [sp.DecodeIds(seq) for seq in seqs]

def __string2sequence(line, EOS_ID, lookup_table):
    tokens = tf.string_split([line]).values 
    ids = tf.cast(lookup_table.lookup(tokens), tf.int32)
    ids = tf.concat([ids, tf.fill([1], EOS_ID)], axis=0)
    ids_lens = (ids, tf.shape(ids)[0])
    return ids_lens

def make_dataset_source_target(
                source_file_name,
                target_file_name,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                shuffle_size=None,
                ncpu=8):
    """load file into dataset"""
    tables = [tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=UNK_ID,
                key_column_index=0)
                for vocab_file_name in (source_vocab_file_name, target_vocab_file_name)]
    source_dataset = tf.data.TextLineDataset(source_file_name)
    target_dataset = tf.data.TextLineDataset(target_file_name)
    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    if shuffle_size is not None:
        dataset = dataset.shuffle(shuffle_size)
    return dataset.map(lambda s,t: (__string2sequence(s, EOS_ID, tables[0]), __string2sequence(t, EOS_ID, tables[1])), ncpu)
    

def make_dataset_single(file_name, vocab_file_name, UNK_ID, EOS_ID, ncpu=8):
    table = tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=UNK_ID,
                key_column_index=0)
    dataset = tf.data.TextLineDataset(file_name)
    return dataset.map(lambda s: __string2sequence(s, EOS_ID, table), ncpu)


def make_dataset_single_from_texts(texts, vocab_file_name, UNK_ID, EOS_ID, ncpu=8):
    table = tf.contrib.lookup.index_table_from_file(
                vocab_file_name,
                num_oov_buckets=0,
                default_value=UNK_ID,
                key_column_index=0)
    dataset = tf.data.Dataset.from_generator(lambda:texts, tf.string, tf.TensorShape([]))
    return dataset.map(lambda s: __string2sequence(s, EOS_ID, table), ncpu)

