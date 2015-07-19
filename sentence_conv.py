from lasagne.base import Layer
from lasagne import nonlinearity
from lasagne import init

class ConvSentence(Layer):
    """
    The sentence must have some fixed length, so we will do some

    """

    def __init__(
            self,
            incoming,
            average=False,
            num_words_per_filter=2,
            num_filters=100,
            nonlinearity=nonlinearity.rectify,
            W=init.GlorotUniform(),
            b=init.Constant(0.),
            max_sentence_length=100,
            **kwargs
    ):
        super(ConvSentence,self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.average = average
        self.num_words_per_filter = num_words_per_filter
        self.nonlinearity = nonlinearity

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        self.b = self.add_param(b, biases_shape, name='b', regularizable=False)

    def get_W_shape(self):
        return (self.num_filters, )  # TODO:


    def get_output_shape_for(self, input_shape):
        # input_shape = (batch_size, . . . )
        return (input_shape[0], self.num_filters)

    def get_output_for(self, input, input_shape=None, **kwargs):
        accumulator = T.matrix()
