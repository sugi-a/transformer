import sys, os, codecs
from logging import getLogger
logger = getLogger()

import tensorflow as tf
from tensorflow.contrib.framework import nest
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
    
def make_batches_source_target_const_capacity_batch_from_list(
                source_list,
                target_list,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID,
                maxlen,
                batch_capacity,
                ncpu=8,
                sort=True,
                allow_skip=True):
    '''
    Args: Mostly the same as make_dataset_source_target.
        maxlen: a sentence pair is ignored if one or both sentence in it has more tokens than `maxlen`
        batch_capacity: batch's capacity (=shape[0]*shape[1]) doesn't exceed this value.
    Returns:
        List of nested structure:
        [batches: (([batch_size: [length: int]], [batch_size: int]),
                    ([batch_size: [length: int]], [batch_size: int]))]
        Samples are sorted by the number of tokens in the source sentences and then batched from the beginning.
        When batching, this method tries to put as many samples as possible into the batch while
        keeping the batch's capacity (shape[0] * shape[1]) less than `batch_capacity`.
        Note that in every training iteration, composition of each batch is invariant: shuffling in
        every training iteration changes the order of batches but the contents in each batch remains the same.
        
    '''
    logger.info('make_batches_source_target_const_capacity_batch_from_list')

    assert maxlen <= batch_capacity

    # Make token->ID mapping
    with codecs.open(source_vocab_file_name) as sv_f, codecs.open(target_vocab_file_name) as tv_f:
        s_token2ID = {line.split()[0]: offset for offset, line in enumerate(sv_f)}
        t_token2ID = {line.split()[0]: offset for offset, line in enumerate(tv_f)}

    # Read lines from the source/target file and zip
    # [dataset size: ([source length: str], [target length: str])]
    zipped_lines = [(sl.strip().split(' '), tl.strip().split(' ')) for sl,tl in zip(source_list, target_list)]

    # Sort by the number of tokens in the source sentence
    if sort:
        zipped_lines.sort(key=lambda x:len(x[0]))

    # Make batches
    logger.info('Making batches. #pairs in the original dataset:{}'.format(len(zipped_lines)))
    batches = []
    s_batch, t_batch, s_lens, t_lens = None, None, None, None
    batch_size = 0 # batch_shape[0]
    batch_length = 1e9 # batch_shzpe[1]
    n_ignored_pairs = 0
    for s_seq, t_seq in zipped_lines:
        # Convert tokens to IDs and EOS
        s_seq = [s_token2ID.get(token, UNK_ID) for token in s_seq if len(token) > 0] + [EOS_ID]
        t_seq = [t_token2ID.get(token, UNK_ID) for token in t_seq if len(token) > 0] + [EOS_ID]

        # get sequence length
        s_len, t_len = len(s_seq), len(t_seq)

        # Skip too long sequences
        if s_len > maxlen or t_len > maxlen or s_len > batch_capacity or t_len > batch_capacity:
            assert allow_skip
            n_ignored_pairs += 1
            continue

        # Update length and size
        batch_length = max(batch_length, s_len, t_len)
        batch_size += 1

        # Make a new minibatch if overflow
        if (batch_size + 1) * batch_length > batch_capacity:
            s_batch, t_batch, s_lens, t_lens = [], [], [], []
            batches.append((s_batch, s_lens, t_batch, t_lens))
            batch_size = 1
            batch_length = max(s_len, t_len)
        s_batch.append(s_seq)
        t_batch.append(t_seq)
        s_lens.append(s_len)
        t_lens.append(t_len)
    logger.info('''Making batches done. Number of ignored pairs:{}
                Number of batches:{}'''.format(n_ignored_pairs, len(batches)))

    # Pad batches. structure: ((seq, len), (seq, len))
    logger.info('Padding batches.')
    padded_batches = []
    for s_batch, s_lens, t_batch, t_lens in batches:
        s_batch_length = max(s_lens)
        t_batch_length = max(t_lens)
        padded_s_batch = [seq + [PAD_ID] * (s_batch_length - l) for seq,l in zip(s_batch, s_lens)]
        padded_t_batch = [seq + [PAD_ID] * (t_batch_length - l) for seq,l in zip(t_batch, t_lens)]
        padded_batches.append(((padded_s_batch, s_lens), (padded_t_batch, t_lens)))
    logger.info('Padding batches done.')

    return padded_batches

def make_dataset_source_target_const_capacity_batch_from_list(
                source_list,
                target_list,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID,
                maxlen,
                batch_capacity,
                ncpu=8,
                sort=True,
                allow_skip=True):
    '''
    Args: Mostly the same as make_dataset_source_target.
        maxlen: a sentence pair is ignored if one or both sentence in it has more tokens than `maxlen`
        batch_capacity: batch's capacity (=shape[0]*shape[1]) doesn't exceed this value.
    Returns:
        Dataset is sorted by the number of tokens in the source sentences and then batched from the beginning.
        When batching, this method tries to put as many samples as possible into the batch while
        keeping the batch's capacity (shape[0] * shape[1]) less than `batch_capacity`.
        Note that in every training iteration, composition of each batch is invariant: shuffling in
        every training iteration changes the order of batches but the contents in each batch remains the same.
        
    '''
    logger.info('make_dataset_source_target_const_capacity_batch_from_list')

    padded_batches = make_batches_source_target_const_capacity_batch_from_list(source_list,
        target_list, source_vocab_file_name, target_vocab_file_name, UNK_ID, EOS_ID, PAD_ID, maxlen,
        batch_capacity, ncpu, sort, allow_skip)

    # Make dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: padded_batches,
        ((tf.int32, tf.int32), (tf.int32, tf.int32)),
        (([None, None], [None]), ([None, None], [None])))
    dataset = dataset.map(lambda s,t: nest.map_structure(lambda x:tf.cast(x, tf.int32), (s, t)))

    return dataset

def make_dataset_source_target_const_capacity_batch(
                source_file_name,
                target_file_name,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID,
                maxlen,
                batch_capacity,
                ncpu=8,
                sort=True,
                allow_skip=True):
    '''
    Args: Mostly the same as make_dataset_source_target.
        maxlen: a sentence pair is ignored if one or both sentence in it has more tokens than `maxlen`
        batch_capacity: batch's capacity (=shape[0]*shape[1]) doesn't exceed this value.
    Returns:
        Dataset is sorted by the number of tokens in the source sentences and then batched from the beginning.
        When batching, this method tries to put as many samples as possible into the batch while
        keeping the batch's capacity (shape[0] * shape[1]) less than `batch_capacity`.
        Note that in every training iteration, composition of each batch is invariant: shuffling in
        every training iteration changes the order of batches but the contents in each batch remains the same.
        
    '''
    logger.info('make_dataset_source_target_const_capacity_batch')

    logger.info('Reading data file.')
    with codecs.open(source_file_name, 'r') as s_f, codecs.open(target_file_name) as t_f:
        source_lines = s_f.readlines()
        target_lines = t_f.readlines()


    return make_dataset_source_target_const_capacity_batch_from_list(souce_lines,
        target_lines, source_vocab_file_name, target_vocab_file_name, UNK_ID,
        EOS_ID, PAD_ID, maxlen, batch_capacity, ncpu, sort, allow_skip)

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
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    return dataset.map(lambda s: __string2sequence(s, EOS_ID, table), ncpu)

