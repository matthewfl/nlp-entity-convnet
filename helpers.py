import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
import theano


class SimpleMaxingLayer(Layer):

    def __init__(self, incoming,  **kwargs):
        super(SimpleMaxingLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        r = (input_shape[0], input_shape[1], 1)
        return r

    def get_output_for(self, input, **kwargs):
        return T.max(input, axis=2)
