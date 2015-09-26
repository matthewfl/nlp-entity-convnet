import numpy as np
import json

class WordVectors(object):

    def __init__(
            self,
            fname,
            redir_fname=None,
            negvectors=False,
            sentence_length=50,
    ):
        self.fname = fname
        self.have_negvectors = negvectors
        self.negvectors = {}
        self.vectors = {}
        self.vector_size = 0
        self.sentence_length = sentence_length
        self.add_unknown_words = True

        self.word_location = {}
        self.word_count = 1
        self.word_matrix = []
        self.reverse_word_location = [None]

        self._load()

        self.word_matrix.append(np.zeros(self.vector_size))
        if redir_fname:
            self.redirects = json.load(open(redir_fname))
        else:
            self.redirects = {}

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

    def _add_unknown_word_rand(self, word):
        "add an unknown word useing a new random vector, good for when training the vectors"
        self.vectors[word] = np.random.uniform(-0.25, 0.25, self.vector_size)
        if self.have_negvectors:
            self.negvectors[word] = np.random.uniform(-0.25, 0.25, self.vector_size)

    def _add_unknown_word(self, word):
        "add a zero vector, good for when there are a lot of these vectors and we want to ignore them"
        self.vectors[word] = np.zeros(self.vector_size)
        if self.have_negvectors:
            self.negvectors[word] = np.zeros(self.vector_size)

    def __getitem__(self, word):
        word = self.redirects.get(word, word)
        r = self.vectors.get(word)
        if r is not None:
            return r
        if self.add_unknown_words:
            self._add_unknown_word(word)
            return self.vectors[word]

    def get_location(self, word):
        r = self.word_location.get(word)
        if r is not None:
            return r
        itm = self[word]
        if itm is None:
            return None
        place = self.word_count
        self.word_count += 1
        self.word_matrix.append(itm)
        self.word_location[word] = place
        self.reverse_word_location.append(word)
        return place

    def get_word(self, location):
        assert location < len(self.reverse_word_location)
        return self.reverse_word_location[location]

    def get_numpy_matrix(self):
        "return a matrix that contains all the words vectors, then can use the tokenized location to lookup a given word"
        return np.array(self.word_matrix)

    def tokenize(self, wrds, length=None):
        ret = []
        if isinstance(wrds, basestring):
            wrds = wrds.lower().split()
        for i in xrange(length or self.sentence_length):
            if i < len(wrds):
                w = self.get_location(wrds[i])
                if w is None:
                    w = 0
                ret.append(w)
            else:
                ret.append(0)
        return ret

class WordTokenizer(object):

    def __init__(self, vectors, random_init=False, add_unknown_words=False, sentence_length=200):
        self.word_vectors = vectors
        if random_init:
            self.new_init = lambda x: np.random.uniform(-0.25, 0.25, x)
        else:
            self.new_init = lambda x: np.zeros(x)
        self.add_unknown_words = add_unknown_words
        self.sentence_length = sentence_length
        
        self.word_location = {}
        self.word_count = 1
        self.word_matrix = []
        self.reverse_word_location = [None]
        self.added_words = {}

    @property
    def vector_size(self):
        return self.word_vectors.vector_size

    def _add_unknown_word(self, word):
        "add a zero vector, good for when there are a lot of these vectors and we want to ignore them"
        r = self.added_words[word] = self.new_init(self.vector_size)
        return r
               
    def __getitem__(self, word):
        word = self.word_vectors.redirects.get(word, word)
        r = self.word_vectors.vectors.get(word)
        if r is not None:
            return r
        if self.add_unknown_words:
            if word in self.added_words:
                return self.added_words[word]
            return self._add_unknown_word(word)
            
    def get_location(self, word):
        r = self.word_location.get(word)
        if r is not None:
            return r
        itm = self[word]
        if itm is None:
            return None
        place = self.word_count
        self.word_count += 1
        self.word_matrix.append(itm)
        self.word_location[word] = place
        self.reverse_word_location.append(word)
        return place

    def get_word(self, location):
        assert location < len(self.reverse_word_location)
        return self.reverse_word_location[location]

    def get_numpy_matrix(self):
        return np.array(self.word_matrix)

    def tokenize(self, wrds, length=None):
        ret = []
        if isinstance(wrds, basestring):
            wrds = wrds.lower().split()
        for i in xrange(length or self.sentence_length):
            if i < len(wrds):
                w = self.get_location(wrds[i])
                if w is None:
                    w = 0
                ret.append(w)
            else:
                ret.append(0)
        return ret



import theano
from lasagne.layers.base import Layer
from lasagne import init

class EmbeddingLayer(Layer):
    """
    lasagne.layers.EmbeddingLayer(incoming, input_size, output_size,
    W=lasagne.init.Normal(), **kwargs)

    A layer for word embeddings. The input should be an integer type
    Tensor variable.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings. The last embedding will have index
        input_size - 1.

    output_size : int
        The size of each embedding.

    W : Theano shared variable, numpy array or callable
        The embedding matrix.

    Examples
    --------
    >>> from lasagne.layers import EmbeddingLayer, InputLayer, get_output
    >>> import theano
    >>> x = T.imatrix()
    >>> l_in = InputLayer((3, ))
    >>> W = np.arange(3*5).reshape((3, 5)).astype('float32')
    >>> l1 = EmbeddingLayer(l_in, input_size=3, output_size=5, W=W)
    >>> output = get_output(l1, x)
    >>> f = theano.function([x], output)
    >>> x_test = np.array([[0, 2], [1, 2]]).astype('int32')
    >>> f(x_test)
    array([[[  0.,   1.,   2.,   3.,   4.],
            [ 10.,  11.,  12.,  13.,  14.]],
    <BLANKLINE>
           [[  5.,   6.,   7.,   8.,   9.],
            [ 10.,  11.,  12.,  13.,  14.]]], dtype=float32)
    """
    def __init__(self, incoming, output_size=None, num_words=None,
                 W=None, add_word_params=False, **kwargs):
        super(EmbeddingLayer, self).__init__(incoming, **kwargs)

        assert (num_words is not None and output_size is not None) or W is not None, 'need the output_shape or W'

        if W is None:
            W = init.Normal()
            self.output_size = output_size
            s = (num_words, output_size)
        else:
            if isinstance(W, theano.compile.SharedVariable):
                s = W.get_value(borrow=True).shape
            else:
                s = W.shape
            self.output_size = s[1]

        if add_word_params:
            self.W = self.add_param(W, s, name="Embeddings",
                                    trainable=add_word_params)
        else:
            if isinstance(W, theano.compile.SharedVariable):
                self.W = W
            else:
                self.W = theano.shared(name='Embeddings', value=W)

    def get_output_shape_for(self, input_shape):
        r = (input_shape[0], 1, input_shape[1], self.output_size)
        return r

    def get_output_for(self, input, **kwargs):
        # if the value at some position is -1, then need to
        return self.W[input].reshape(self.get_output_shape_for(input.shape))
