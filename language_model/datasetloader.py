import sys, random
from collections import deque


class RandomSlidingWindow:
    def __init__(self, files, vocab, window_size, drop_remainder=True, random=True):
        self.files = files
        self.vocab = vocab
        self.window_size = window_size
        self.drop_remainder = drop_remainder
        self.random = random


    def __iter__(self):
        return self.gen()


    def __call__(self):
        return self.gen()


    def gen(self):
        files = sorted(self.files)
        
        for fn in files:
            with open(fn) as f:
                q = deque()

                # if self.random, window size is randomly initialized
                win_size = random.randint(1, self.window_size) if self.random else self.window_size
                for line in f:
                    # Check the document boundaries
                    if len(line) == 1:
                        if (not self.drop_remainder) and len(q) >= 50:
                            yield list(q)

                        q.clear()
                    else:
                        q.extend(self.vocab.line2IDs(line, False))

                        while len(q) >= self.window_size:
                            popped = [q.popleft() for i in range(win_size)]
                            if win_size != self.window_size:
                                if (not self.drop_remainder) and len(q) >= 50:
                                    yield popped
                            else:
                                yield popped

                            # win_size == self.window_size for now
                            win_size = self.window_size

                # Yield the remainder if it should be
                if (not self.drop_remainder) and len(q) >= 50:
                    yield list(q)

