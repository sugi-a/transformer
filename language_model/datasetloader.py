import sys
from collections import deque

class RandomSlidingWindow:
    def __init__(self, files, vocab, window_size, drop_remainder=True):
        self.files = files
        self.vocab = vocab
        self.window_size = window_size
        self.drop_remainder = drop_remainder


    def __iter__(self):
        return self.gen()


    def gen(self):
        files = sorted(self.files)
        
        for fn in files:
            with open(fn) as f:
                q = deque()
                for line in f:
                    # Check the document boundaries
                    if len(line) == 1:
                        if (not self.drop_remainder) and len(q) >= self.window_size // 2:
                            yield list(q)

                        q.clear()
                    else:
                        q.extend(vocab.line2IDs(line, False))

                        while len(q) >= self.window_size:
                            yield [q.popleft() for i in range(self.window_size)]

                # Yield the remainder if it should be
                if (not self.drop_remainder) and len(q) >= self.window_size // 2:
                    yield list(q)

