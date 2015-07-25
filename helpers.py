import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
import theano


class SimpleMaxingLayer(Layer):

    def __init__(self, incoming, axis=2, **kwargs):
        super(SimpleMaxingLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        r = list(input_shape)
        del r[self.axis]
        return tuple(r)

    def get_output_for(self, input, **kwargs):
        return T.max(input, axis=self.axis)
