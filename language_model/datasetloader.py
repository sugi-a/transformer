import sys, random, json, random
from logging import getLogger; logger = getLogger(__name__)
from collections import deque


class RandomSlidingWindow:
    def __init__(self, files, vocab, window_size, keep_remainder_larger_equal=None, random=True, state_log_file=None):
        self.files = files
        self.vocab = vocab
        self.window_size = window_size
        self.keep_rem_le = window_size if keep_remainder_larger_equal is None else \
            min(window_size, max(1, keep_remainder_larger_equal))
        self.random = random
        self.state_log_file = state_log_file

        if self.state_log_file:
            try:
                # Load the existing one
                with open(self.state_log_file) as f:
                    self.state = json.load(f)

                logger.info('State log file was found.')

                # Check file list agreement
                if self.state['files'] != files:
                    logger.error('The file list differs from the existing one in the state log. Delete the state log file if you update the file list.')
                    exit(1)
            except FileNotFoundError:
                logger.info('State log file was not found.')
                # Newly create
                self.state = {'files': self.files}
        else:
            self.state = {'files': self.files}
            

    def save_state(self):
        if not self.state_log_file:
            return

        with open(self.state_log_file, 'w') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=4)


    def __iter__(self):
        return self.gen()


    def __call__(self):
        return self.gen()


    def gen(self):

        # Initialize file list
        if not self.state.get('shuffled_files', None):
            self.state['shuffled_files'] = random.sample(self.files, len(self.files))

        # Initialize count
        if self.state.get('data_count', None) is None:
            self.state['data_count'] = 0
        else:
            logger.debug('First {} data will be skipped'.format(self.state['data_count']))

        start_data_count = self.state['data_count']
        _skip_count = 0

        for fn in self.state['shuffled_files']:
            logger.debug('Open data file: {}'.format(fn))
            with open(fn) as f:
                q = deque()

                # if self.random, window size is randomly initialized
                win_size = random.randint(1, self.window_size) if self.random else self.window_size
                for line in f:
                    # Check the document boundaries
                    if len(line) == 1:
                        popped = list(q)
                        q.clear()
                        if len(popped) >= self.keep_rem_le:
                            #### SKIP OR YIELD ####
                            if _skip_count < start_data_count:
                                _skip_count += 1
                            else:
                                yield popped
                                self.state['data_count'] += 1
                            #######################
                    else:
                        q.extend(self.vocab.line2IDs(line, False))

                        while len(q) >= self.window_size:
                            popped = [q.popleft() for i in range(win_size)]
                            if len(popped) >= self.keep_rem_le:
                                #### SKIP OR YIELD ####
                                if _skip_count < start_data_count:
                                    _skip_count += 1
                                else:
                                    yield popped
                                    self.state['data_count'] += 1
                                #######################

                            # win_size == self.window_size for now
                            win_size = self.window_size

                # Yield the remainder if it should be
                popped = list(q)
                if len(popped) >= self.keep_rem_le:
                    #### SKIP OR YIELD ####
                    if _skip_count < start_data_count:
                        _skip_count += 1
                    else:
                        yield popped
                        self.state['data_count'] += 1
                    #######################

            # Save state
            if self.state_log_file:
                self.save_state()

        
        # Reset state
        self.state['shuffled_files'] = None
        self.state['data_count'] = None
        


def make_const_capacity_batch_list(data, lengths, capacity, PAD_ID):
    assert all(l < capacity for l in lengths)

    ret = []
    batch = []
    batch_len = []

    mlen = 0
    for d, l in zip(data, lengths):
        if max(mlen, l) * (len(batch) + 1) > capacity:
            if batch:
                ret.append((batch, batch_len))
                batch = []
                batch_len = []
                mlen = 0

        mlen = max(mlen, l)

        batch.append(d)
        batch_len.append(l)

    if batch:
        ret.append((batch, batch_len))

    # Padding
    for batch, batch_len in ret:
        maxlen = max(batch_len)
        for i in range(len(batch)):
            batch[i] = batch[i] + [PAD_ID] * (maxlen - len(batch[i]))

    return ret

