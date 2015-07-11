import numpy as np


class WordVectors(object):

    def __init__(
            self,
            fname,
            negvectors=False
    ):
        self.fname = fname
        self.have_negvectors = negvectors
        self.negvectors = {}
        self.vectors = {}
        self.vector_size = 0
        self._load()

    def _load(self):
        # based off the loader from CNN_sentence
        with open(self.fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            self.vector_size = layer1_size
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                self.vectors[word] = np.fromstring(f.read(binary_len), dtype='float32')
                if self.have_negvectors:
                    self.negvectors[word] = np.fromstring(f.read(binary_len), dtype='float32')

    def _add_unknown_word(self, word):
        self.vectors[word] = np.random.uniform(-0.25, 0.25, self.vector_size)
        if self.have_negvectors:
            self.negvectors[word] = np.random.uniform(-0.25, 0.25, self.vector_size)

    def __getitem__(self, word):
        r = self.vectors.get(word)
        if r is not None:
            return r
        self._add_unknown_word(word)
        return self.vectors[word]
