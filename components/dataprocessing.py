import sys, os, codecs, time
from logging import getLogger
logger = getLogger(__name__)

import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np


class Vocabulary:
    def __init__(self, vocab_file, PAD_ID, EOS_ID, UNK_ID, other_control_symbols=None):
        with open(vocab_file, 'r') as f:
            self.ID2tok = [line.split()[0] for line in f]
            self.tok2ID = {tok: i for i, tok in enumerate(self.ID2tok)}

        self.UNK_ID = UNK_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.ctrls = set(other_control_symbols or []) | {PAD_ID, EOS_ID}


    def line2IDs(self, line):
        return [self.tok2ID.get(tok, self.UNK_ID) for tok in line] + [self.EOS_ID]


    def text2IDs(self, text):
        return list(map(self.line2ID, text))


    def IDs2text(self, IDs):
        return [' '.join(self.ID2tok[id]
            for id in sent if not id in self.ctrls) for sent in IDs]



def __string2sequence(line, EOS_ID, lookup_table):
    tokens = tf.string_split([line]).values 
    ids = tf.cast(lookup_table.lookup(tokens), tf.int32)
    if EOS_ID is not None:
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
    if type(source_file_name) == type([]):
        assert type(target_file_name)==type([]) and len(target_file_name)==len(source_file_name)
        source_fnames = tf.data.Dataset.from_tensor_slices(source_file_name)
        target_fnames = tf.data.Dataset.from_tensor_slices(target_file_name)
        dataset = tf.data.Dataset.zip((source_fnames, target_fnames))
        dataset = dataset.shuffle(len(source_file_name))
        dataset = dataset.flat_map(lambda sn,tn:
            tf.data.Dataset.zip(tuple(tf.data.TextLineDataset(fname) for fname in [sn,tn])))
    else:
        source_dataset = tf.data.TextLineDataset(source_file_name)
        target_dataset = tf.data.TextLineDataset(target_file_name)
        dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    if shuffle_size is not None:
        dataset = dataset.shuffle(shuffle_size)
    return dataset.map(lambda s,t: (__string2sequence(s, EOS_ID, tables[0]), __string2sequence(t, EOS_ID, tables[1])), ncpu)

def make_source_target_zipped_list(
                source_list,
                target_list,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID):
    '''
    Args: Mostly the same as make_dataset_source_target.
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
    _start_time = time.time()
    logger.debug('make_source_target_zipped_list')

    # Make token->ID mapping
    with codecs.open(source_vocab_file_name) as sv_f, codecs.open(target_vocab_file_name) as tv_f:
        s_token2ID = {line.split()[0]: offset for offset, line in enumerate(sv_f)}
        t_token2ID = {line.split()[0]: offset for offset, line in enumerate(tv_f)}

    # Read lines from the source/target file and zip
    # [dataset size: ([source length: str], [target length: str])]
    zipped_lines = [(sl.strip().split(' '), tl.strip().split(' ')) for sl,tl in zip(source_list, target_list)]

    # Make batches
    logger.debug('#pairs in the original dataset:{}'.format(len(zipped_lines)))

    # Convert to ID, add EOS and check length
    new_zipped_lines = []
    for s_seq, t_seq in zipped_lines:
        # Convert tokens to IDs
        s_seq = [s_token2ID.get(token, UNK_ID) for token in s_seq if len(token) > 0]
        t_seq = [t_token2ID.get(token, UNK_ID) for token in t_seq if len(token) > 0]

        # Add EOS
        if EOS_ID is not None:
            s_seq = s_seq + [EOS_ID]
            t_seq = t_seq + [EOS_ID]

        new_zipped_lines.append((s_seq, t_seq))

    logger.debug('make_source_target_zipped_list')
    logger.debug('''make_source_target_zipped_list done. {}sec'''.format(time.time() - _start_time))

    return new_zipped_lines

def make_batches_from_zipped_list(
                zipped_lines,
                PAD_ID,
                batch_capacity,
                order_mode=None,
                allow_skip=False):

    _start_time = time.time()

    if order_mode == 'sort':
        zipped_lines.sort(key=lambda x: len(x[0]))
        # avoid containing equivalent sentences in a batch
        # by disperating them to four distant positions in the dataset
        zipped_lines = sum((zipped_lines[i::4] for i in range(4)), [])

    elif order_mode == 'shuffle':
        zipped_lines = np.random.permutation(zipped_lines)
    else:
        assert order_mode is None

    batches = []
    s_batch, t_batch, s_lens, t_lens = None, None, None, None
    batch_size = 0 # batch_shape[0]
    batch_length = 1e9 # batch_shzpe[1]
    n_ignored_pairs = 0
    for s_seq, t_seq in zipped_lines:
        # get sequence length
        s_len, t_len = len(s_seq), len(t_seq)

        # Skip too long sequences
        if s_len > batch_capacity or t_len > batch_capacity:
            assert allow_skip
            n_ignored_pairs += 1
            continue

        # Update length and size
        batch_length = max(batch_length, s_len, t_len)
        batch_size += 1

        # Make a new minibatch if overflow
        if batch_size * batch_length > batch_capacity:
            s_batch, t_batch, s_lens, t_lens = [], [], [], []
            batches.append((s_batch, s_lens, t_batch, t_lens))
            batch_size = 1
            batch_length = max(s_len, t_len)
        s_batch.append(s_seq)
        t_batch.append(t_seq)
        s_lens.append(s_len)
        t_lens.append(t_len)
    logger.debug('''Making batches done. Number of ignored pairs:{}
Number of batches:{}, time:{}sec'''.format(
    n_ignored_pairs, len(batches), time.time()-_start_time))

    # Pad batches. structure: ((seq, len), (seq, len))
    _start_time = time.time()
    logger.debug('Padding batches.')
    padded_batches = []
    for s_batch, s_lens, t_batch, t_lens in batches:
        s_batch_length = max(s_lens)
        t_batch_length = max(t_lens)
        padded_s_batch = [seq + [PAD_ID] * (s_batch_length - l) for seq,l in zip(s_batch, s_lens)]
        padded_t_batch = [seq + [PAD_ID] * (t_batch_length - l) for seq,l in zip(t_batch, t_lens)]
        padded_batches.append(((padded_s_batch, s_lens), (padded_t_batch, t_lens)))
    logger.debug('Padding batches done. time: {}sec'.format(time.time() - _start_time))

    if order_mode == 'sort':
        logger.debug('permutation of sorted batches')
        np.random.shuffle(padded_batches)
    return padded_batches
    
    
def make_batches_source_target_const_capacity_batch_from_list(
                source_list,
                target_list,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID,
                batch_capacity,
                order_mode=None,
                allow_skip=False):
    '''
    Args: Mostly the same as make_dataset_source_target.
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
    logger.debug('make_batches_source_target_const_capacity_batch_from_list')

    if type(source_list) == 'str': source_list = [source_list]
    if type(target_list) == 'str': target_list = [target_list]
    assert len(source_list) == len(target_list)
    zipped_list = make_source_target_zipped_list(source_list, target_list,
        source_vocab_file_name, target_vocab_file_name,
        UNK_ID, EOS_ID, PAD_ID)

    # Make batches
    padded_batches = make_batches_from_zipped_list(zipped_list, PAD_ID, batch_capacity,
        order_mode, allow_skip)

    return padded_batches

def make_dataset_source_target_const_capacity_batch_from_list(
                source_list,
                target_list,
                source_vocab_file_name,
                target_vocab_file_name,
                UNK_ID,
                EOS_ID,
                PAD_ID,
                batch_capacity,
                order_mode=None,
                allow_skip=False):
    '''
    Args: Mostly the same as make_dataset_source_target.
        batch_capacity: batch's capacity (=shape[0]*shape[1]) doesn't exceed this value.
    Returns:
        Dataset is sorted by the number of tokens in the source sentences and then batched from the beginning.
        When batching, this method tries to put as many samples as possible into the batch while
        keeping the batch's capacity (shape[0] * shape[1]) less than `batch_capacity`.
        Note that in every training iteration, composition of each batch is invariant: shuffling in
        every training iteration changes the order of batches but the contents in each batch remains the same.
        
    '''
    logger.debug('make_dataset_source_target_const_capacity_batch_from_list')

    zipped_list = make_source_target_zipped_list(source_list, target_list,
        source_vocab_file_name, target_vocab_file_name,
        UNK_ID, EOS_ID, PAD_ID)

    # Make batches
    def gen():
        return make_batches_from_zipped_list(zipped_list,
            PAD_ID, batch_capacity, order_mode, allow_skip)

    # Make dataset
    dataset = tf.data.Dataset.from_generator(
        gen,
        ((tf.int32, tf.int32), (tf.int32, tf.int32)),
        (([None, None], [None]), ([None, None], [None])))
    dataset = dataset.map(lambda s,t: nest.map_structure(lambda x:tf.cast(x, tf.int32), (s, t)))

    return dataset

def make_dataset_source_target_const_capacity_batch(
                source_file_name,
                target_file_name,
                *args, **kwargs):
    '''
    Args: Mostly the same as make_dataset_source_target.
        batch_capacity: batch's capacity (=shape[0]*shape[1]) doesn't exceed this value.
    Returns:
        Dataset is sorted by the number of tokens in the source sentences and then batched from the beginning.
        When batching, this method tries to put as many samples as possible into the batch while
        keeping the batch's capacity (shape[0] * shape[1]) less than `batch_capacity`.
        Note that in every training iteration, composition of each batch is invariant: shuffling in
        every training iteration changes the order of batches but the contents in each batch remains the same.
        
    '''
    logger.debug('make_dataset_source_target_const_capacity_batch')

    logger.debug('Reading data file.')
    if type(source_file_name) == type([]):
        assert len(source_file_name) == len(target_file_name)
        source_lines, target_lines = [], []
        for sfn, tfn in zip(source_file_name, target_file_name):
            with codecs.open(sfn, 'r') as s_f, codecs.open(tfn, 'r') as t_f:
                source_lines.extend(s_f.readlines())
                target_lines.extend(t_f.readlines())
    else:
        with codecs.open(source_file_name, 'r') as s_f, codecs.open(target_file_name) as t_f:
            source_lines = s_f.readlines()
            target_lines = t_f.readlines()

    return make_dataset_source_target_const_capacity_batch_from_list(
        source_lines,
        target_lines,
        *args, **kwargs)

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

