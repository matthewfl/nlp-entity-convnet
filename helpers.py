import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer, MergeLayer
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


class SimpleAverageLayer(MergeLayer):
    "compute the length of sentence and use that for the average instead of the vector length"

    def __init__(self, incomings, axis=2, **kwargs):
        super(SimpleAverageLayer, self).__init__(incomings, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        r = list(input_shapes[0])
        del r[self.axis]
        return tuple(r)

    def get_output_for(self, inputs, **kwargs):
        sent_length = T.sum(T.neq(inputs[1], 0), axis=1)
        sums = T.sum(inputs[0], axis=self.axis)
        return sums / sent_length
