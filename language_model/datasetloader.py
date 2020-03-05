import sys, json, random
from logging import getLogger; logger = getLogger(__name__)
from collections import deque

from ..components.dataprocessing import gen_json_resumable

class MultiSentenceSlidingWindowLoader:
    def __init__(self, files, vocab, window_size, keep_remainder_larger_equal=None,
        random=True, state_log_file=None,
        header = None):

        self.files = files
        self.vocab = vocab
        self.window_size = window_size
        self.keep_rem_le = keep_remainder_larger_equal
        self.random = random
        self.state_log_file = state_log_file
        if header is None:
            self.header = None
        else:
            if type(header) == int:
                self.header = header
            elif type(header) == str:
                self.header = self.vocab.tok2ID[header]
            else:
                raise ValueError

    def __call__(self):
        return self.gen()


    def gen(self):
        files = random.sample(self.files, len(self.files))
        if self.state_log_file:
            files = gen_json_resumable(files, self.state_log_file)
        for fn in files:
            logger.debug('Opening file {}'.format(fn))
            with open(fn) as f:
                q = deque()
                win_size = random.randint(1, self.window_size) if self.random else self.window_size
                for line in f:
                    # Check the document boundaries
                    if len(line) == 1:
                        if self.header is not None:
                            q.appendleft(self.header)
                        popped = list(q)
                        q.clear()
                        if len(popped) >= self.keep_rem_le:
                            yield popped
                    else:
                        q.extend(self.vocab.line2IDs(line))

                        while len(q) >= self.window_size:
                            if self.header is not None:
                                q.appendleft(self.header)
                            popped = [q.popleft() for i in range(win_size)]
                            if len(popped) >= self.keep_rem_le:
                                yield popped
                            win_size = self.window_size

                # Yield the remainder
                if self.header is not None:
                    q.appendleft(self.header)
                popped = list(q)
                if len(popped) >= self.keep_rem_le:
                    yield popped

