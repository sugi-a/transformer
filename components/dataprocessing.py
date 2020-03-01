import sys, os, codecs, time, heapq, random, time, json
from contextlib import ExitStack
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


    def tokens2IDs(self, tokens, put_sos, put_eos):
        ret = [self.tok2ID.get(tok, self.UNK_ID) for tok in tokens]
        if put_eos:
            ret = ret + [self.EOS_ID]
        if put_sos:
            ret = [self.SOS_ID] + ret
        return ret


    def line2IDs(self, line, put_sos, put_eos):
        return self.tokens2IDs(line.split(), put_sos=put_sos, put_eos=put_eos)


    def text2IDs(self, text, put_sos, put_eos):
        return [self.line2IDs(line, put_sos=put_sos, put_eos=put_eos) for line in text]


    def IDs2text(self, IDs, skip_control_symbols=True):
        if skip_control_symbols:
            return [' '.join(self.ID2tok[id]
                for id in sent if not id in self.ctrls) for sent in IDs]
        else:
            return [' '.join(self.ID2tok[id]
                for id in sent) for sent in IDs]


def gen_line2IDs(line_iter, vocab, put_sos, put_eos):
    for line in line_iter:
        yield vocab.line2IDs(line, put_eos=put_eos, put_sos=put_sos)


def list2numpy_nested(nested):
    """Converts lists in the nested structure of dict and tuple into numpy arrays"""
    if isinstance(nested, dict):
        new = {k: list2numpy_nested(v) for k, v in nested.items()}
    elif isinstance(nested, tuple):
        new = tuple(list2numpy_nested(v) for v in nested)
    elif isinstance(nested, list):
        new = np.array(nested)
    else:
        new = nested

    return new


def gen_list2numpy_nested(nested_iter):
    for nested in nested_iter:
        yield list2numpy_nested(nested)


def pad_seqs(seqs, maxlen=None, PAD_ID=0):
    maxlen = maxlen or max(len(seq) for seq in seqs)
    return [seq + [PAD_ID] * (maxlen - len(seq)) for seq in seqs]


def gen_multi_padded_batch(multi_seq_iter, batch_size, PAD_ID=0):
    batches = []
    for items in multi_seq_iter:
        if len(batches) == batch_size:
            # [batch, nitmes]-> [nitems, batch]
            yield tuple(
                (pad_seqs(seqs, PAD_ID=PAD_ID), [len(seq) for seq in seqs])
                for seqs in zip(*batches))
            batches = []

        batches.append(items)

    if len(batches) > 0:
        yield tuple(
            (pad_seqs(seqs, PAD_ID=PAD_ID), [len(seq) for seq in seqs])
            for seqs in zip(*batches))

def gen_const_capacity_batch(seq_iter, capacity, PAD_ID=0):
    seqs = []
    lens = []
    maxlen = 0
    for seq in seq_iter:
        l = len(seq)
        if l > capacity:
            raise(ValueError, 'Sequence longer than batch capacity. ({} vs {})'.format(l, capacity))
        
        batch_len = len(seqs)
        if max(maxlen, l) * (batch_len + 1) > capacity:
            yield (pad_seqs(seqs, maxlen=maxlen, PAD_ID=PAD_ID), lens)
            seqs = []
            lens = []
            maxlen = 0

        maxlen = max(maxlen, l)
        seqs.append(seq)
        lens.append(l)
    
    if len(seqs) > 0:
        yield pad_seqs(seqs, maxlen=maxlen, PAD_ID=PAD_ID), lens


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
    
    seqs1, seqs2 = [], []
    lens1, lens2 = [], []
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
        
            seqs1, seqs2 = [], []
            lens1, lens2 = [], []
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


def gen_length_smooth_sorted_seq(iters, buffer_size=10000, nbins=250):
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


def gen_random_sample(iterable, bufsize=None):
    if bufsize is None:
        yield from (iterable)
    else:
        buf = []
        for x in iterable:
            if len(buf) < bufsize:
                buf.append(x)
            else:
                ind = random.randint(0, bufsize - 1)
                yield buf[ind]
                buf[ind] = x
        
        random.shuffle(buf)
        yield from buf


def gen_segment_sort(iterable, segsize=10000, key=None):
    key = key or (lambda x:len(x[0]))
    seg = []
    for x in iterable:
        seg.append(x)
        if len(seg) >= segsize:
            seg.sort(key=key)
            yield from seg
            seg.clear()
    seg.sort()
    yield from seg

            
def gen_lines_from_files(file_names, shuffle=False):
    if shuffle:
        file_names = list(file_names)
        yield from random.sample(file_name, len(file_names))
    for fname in file_names:
        with open(fname) as f:
            for line in f:
                yield line


def gen_lines_from_files_multi(multi_file_names):
    """
    multi_file_names: [(src_file1, trg_file1), (src_file2, trg_file2), ...]
        """
    for fnames in multi_file_names:
        logger.debug('Opening files: {}'.format(', '.join(fnames)))
        with ExitStack() as stack:
            fps = [stack.enter_context(open(fname)) for fname in fnames]
            yield from zip(*fps)


def gen_lines_from_file(file_name):
    with open(file_name) as f:
        for line in f:
            yield line


class CallGenWrapper:
    def __init__(self, init_generator_fn):
        self.init_generator_fn = init_generator_fn
        self.funcs = []
    
    def map(self, fn, *args, **kwargs):
        self.funcs.append((fn , args, kwargs))
        return self


    def map_element(self, fn):
        self.funcs.append((lambda it: map(fn, it), [], {}))
        return self 

    def __call__(self):
        gen = self.init_generator_fn()
        for fn, args, kwargs in self.funcs:
            gen = fn(gen, *args, **kwargs)
        
        return gen

    
    @classmethod
    def zip(cls, *wrappers):
        return cls(lambda: zip(*(wrapper() for wrapper in wrappers)))


def gen_json_resumable(json_iter, state_file, allow_sorting_match=True):
    """Placed in a pipeline, this generator behave like a checkpoint.
    Args:
        json_iter: list of objects which can be converted into json
        """
    
    obj_list = json.loads(json.dumps(list(json_iter)))
    try:
        with open(state_file) as f:
            state = json.load(f)

        if (allow_sorting_match and sorted(obj_list) != sorted(state['obj_list'])) or \
            (not allow_sorting_match and obj_list != state['obj_list']):
            raise Exception('Specified object list differs from the saved one.')

        if len(state['obj_list']) <= state['current']:
            state = None
    except FileNotFoundError:
        state = None

    if state is None:
        state = {'obj_list': obj_list, 'current': 0}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)

    while state['current'] < len(state['obj_list']):
        yield state['obj_list'][state['current']]
        
        state['current'] += 1
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)

