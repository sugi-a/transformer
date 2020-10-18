from logging import getLogger; logger = getLogger(__name__)


class Vocabulary:
    def __init__(
            self,
            vocab_file,
            PAD_ID, EOS_ID, UNK_ID,
            SOS_ID=None,
            other_control_symbols=None):
        with open(vocab_file, 'r') as f:
            self.ID2tok = [line.split()[0] for line in f]
            self.tok2ID = {tok: i for i, tok in enumerate(self.ID2tok)}

        self.UNK_ID = UNK_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.SOS_ID = SOS_ID
        self.ctrls = set(other_control_symbols or []) | {PAD_ID, EOS_ID, SOS_ID}


    def tokens2IDs(self, tokens):
        return [self.tok2ID.get(tok, self.UNK_ID) for tok in tokens]


    def line2IDs(self, line):
        return self.tokens2IDs(line.split())


    def text2IDs(self, text):
        return [self.line2IDs(line) for line in text]


    def IDs2text(self, IDs, skip_control_symbols=True):
        if skip_control_symbols:
            return [' '.join(self.ID2tok[id]
                for id in sent if not id in self.ctrls) for sent in IDs]
        else:
            return [' '.join(self.ID2tok[id]
                for id in sent) for sent in IDs]

