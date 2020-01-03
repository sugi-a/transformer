import sys, os, codecs, time
from collections import deque
from logging import getLogger; logger = getLogger(__name__)


class Vocabulary:
    def __init__(self, vocab_file, PAD_ID, EOS_ID, UNK_ID, SOS=None, other_control_symbols=None):
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


def pad_seqs(seqs, maxlen=None, PAD_ID=0):
    maxlen = maxlen or 
def const_capacity_batch_generator(text_iter, capacity, vocab, put_eos=False, put_sos=True):
    seqs = deque()
    lens = deque()
    maxlen = 0
    for line in zip(text_iter):
        IDs = vocab.line2IDs(line, put_eos, put_sos)
        l = len(IDs)
        if l > capacity:
            raise(ValueError, 'Sentence () longer than batch capacity. ({} vs {})'.format(line[:20], l, capacity))
        
        if max(maxlen, l) * (len(seqs) + 1) > capacity and len(seqs) > 0:
            yield seqs
            seqs = deque()
            lens = deque()
            maxlen = 0

        maxlen = max(maxlen, l)
        seqs.append(IDs)
        lens.append(l)

        


def xyz():
    with open(fname) as f:
        const_capacity_batch_generator(f, )
