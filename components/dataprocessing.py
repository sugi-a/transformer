import sys, os, codecs, time, heapq, random, time
from collections import deque
from logging import getLogger; logger = getLogger(__name__)
import numpy as np

class Vocabulary:
    def __init__(self, vocab_file, PAD_ID, EOS_ID, UNK_ID, SOS_ID=None, other_control_symbols=None):
        with open(vocab_file, 'r') as f:
            self.ID2tok = [line.split()[0] for line in f]
            self.tok2ID = {tok: i for i, tok in enumerate(self.ID2tok)}

        self.UNK_ID = UNK_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.SOS_ID = SOS_ID
        self.ctrls = set(other_control_symbols or []) | {PAD_ID, EOS_ID, SOS_ID}


    def tokens2IDs(self, tokens, put_eos=True, put_sos=False):
        ret = [self.tok2ID.get(tok, self.UNK_ID) for tok in tokens]
        if put_eos:
            ret = ret + [self.EOS_ID]
        if put_sos:
            ret = [self.SOS] + ret
        return ret


    def line2IDs(self, line, put_eos=True, put_sos=False):
        return self.tokens2IDs(line.split(), put_eos, put_sos)


    def text2IDs(self, text, put_eos=True, put_sos=False):
        return [self.line2IDs(line, put_eos, put_sos) for line in text]


    def IDs2text(self, IDs):
        return [' '.join(self.ID2tok[id]
            for id in sent if not id in self.ctrls) for sent in IDs]


def gen_line2IDs(line_iter, vocab, put_eos=True, put_sos=False):
    for line in line_iter:
        yield vocab.line2IDs(line)

def pad_seqs(seqs, maxlen=None, PAD_ID=0):
    maxlen = maxlen or max(len(seq) for seq in seqs)
    return [seq + [PAD_ID] * (maxlen - len(seq)) for seq in seqs]


def gen_const_capacity_batch(seq_iter, capacity, PAD_ID=0):
    seqs = deque()
    lens = deque()
    maxlen = 0
    for seq in zip(seq_iter):
        l = len(seq)
        if l > capacity:
            raise(ValueError, 'Sequence longer than batch capacity. ({} vs {})'.format(l, capacity))
        
        batch_len = len(seqs)
        if max(maxlen, l) * (batch_len + 1) > capacity:
            yield (pad_seqs(seqs, maxlen=maxlen, PAD_ID=PAD_ID), lens)
            seqs = deque()
            lens = deque()
            maxlen = 0

        maxlen = max(maxlen, l)
        seqs.append(seq)
        lens.append(l)
    
    if len(seqs) > 0:
        yield pad_seqs(seqs, maxlen=maxlen, PAD_ID=PAD_ID)


def gen_dual_const_capacity_batch(dual_seq_iter, capacity, PAD_ID=0):
    """Create paired batches. ((batch1, lengths1), (batch2, lengths2))
    Each batch does not exceed `capacity` in (batch size) x (max length).
    Therefore, total number of tokens the batches can be up to 2 x `capacity`.
    Args:
        dual_seq_iter: iterator which gives (seq1, seq2) at each call
    Returns:
        ((batch1, lengths1), (batch2, lengths2))
        batch_i: Shape [batch_size, maxlength_i], DType int
        lengths_i: Shape [batch_size], DType int
    """
    
    seqs1, seqs2 = deque(), deque()
    lens1, lens2 = deque(), deque()
    maxlen1, maxlen2 = 0, 0

    for seq1, seq2 in dual_seq_iter:
        l1, l2 = len(seq1), len(seq2)
        if l1 > capacity or l2 > capacity:
            raise(ValueError, 'Sequence longer than batch capacity. (seq1: {}, seq2: {}, batch capacity: {})'.format(l1, l2, capacity))

        batch_len = len(seqs1)
        if max(maxlen1, l1) * (batch_len + 1) > capacity or max(maxlen2, l2) * (batch_len + 1) > capacity:
            padded1 = pad_seqs(seqs1, maxlen=maxlen1, PAD_ID=PAD_ID)
            padded2 = pad_seqs(seqs2, maxlen=maxlen2, PAD_ID=PAD_ID)
            yield ((padded1, lens1), (padded2, lens2))
        
            seqs1, seqs2 = deque(), deque()
            lens1, lens2 = deque(), deque()
            maxlen1, maxlen2 = 0, 0
        
        maxlen1, maxlen2 = max(maxlen1, l1), max(maxlen2, l2)
        seqs1.append(seq1)
        seqs2.append(seq2)
        lens1.append(l1)
        lens2.append(l2)

    if len(seqs1) > 0:
        padded1 = pad_seqs(seqs1, maxlen=maxlen1, PAD_ID=PAD_ID)
        padded2 = pad_seqs(seqs2, maxlen=maxlen2, PAD_ID=PAD_ID)
        yield ((padded1, lens1), (padded2, lens2))


def gen_length_smooth_sorted_seq(iters, buffer_size=10000, nbins=300):
    """
    Args:
        iters: an iterator which gives (sequence iterator, iter2, iter3,...) at each call
    """
    bins = [deque() for i in range(nbins)]
    last_len = 0

    nitems = 0
    _start_t = time.time()
    for items in iters:
        if nitems == buffer_size:
            logger.debug('len_smooth_sort buffer filled. size: {}, #bins: {}, time:{}'
                .format(buffer_size, nbins, time.time() - _start_t))
        if nitems >= buffer_size:
            for i in range(max(last_len + 1, nbins - last_len)):
                i_u = last_len + i
                if i_u < nbins and len(bins[i_u]) > 0:
                    yield bins[i_u].pop()
                    last_len = i_u
                    break
                i_d = last_len - i
                if i_d >= 0 and len(bins[i_d]) > 0:
                    yield bins[i_d].pop()
                    last_len = i_d
                    break
        
        bins[min(nbins - 1, len(items[0]))].append(items)
        nitems += 1
    
    for b in bins:
        while len(b) > 0:
            yield b.pop()
            
            
def gen_lines_from_files(file_names, shuffle=False):
    if shuffle:
        file_names = np.random.permutation(file_names)
    for fname in file_names:
        with open(fname) as f:
            for line in f:
                yield line


def gen_lines_from_file(file_name):
    with open(file_name) as f:
        for line in f:
            yield line


def gen_random_sample(iters, buffer_size):
    buffer = []
    
    for items in iters:
        if len(buffer) < buffer_size:
            buffer.append(items)
        else:
            ind = random.randint(0, buffer_size - 1)
            yield buffer[ind]
            buffer[ind] = items
    
    for i in range(len(buffer) - 1, -1, -1):
        ind = random.randint(0, i)
        yield buffer[ind]
        buffer[ind] = buffer[i]


class CallGenWrapper:
    def __init__(self, init_generator_fn):
        self.init_generator_fn = init_generator_fn
        self.funcs = []
    
    def map(self, fn, *args, **kwargs):
        self.funcs.append((fn , args, kwargs))
        return self
    

    def __call__(self):
        gen = self.init_generator_fn()
        for fn, args, kwargs in self.funcs:
            gen = fn(gen, *args, **kwargs)
        
        return gen

    
    @classmethod
    def zip(cls, *wrappers):
        return cls(lambda: zip(*(wrapper() for wrapper in wrappers)))
