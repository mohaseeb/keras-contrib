from keras.engine import Layer
from keras.engine import InputSpec
from .. import backend as K

import numpy as np


class BufferLayer(Layer):
    # FIXME DOCS

    def __init__(self, buffer_size, **kwargs):
        super(BufferLayer, self).__init__(**kwargs)
        self.buffer_size = buffer_size
        self.input_dim = None
        self.batch_size = None
        self.buffer = None
        self.stateful = True  # TODO support non-stateful

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.input_dim = input_shape[-1]  # FIXME the input could multi-dim

        if self.batch_size is None:
            raise ValueError('the batch size needs to be set for this layer. '
                             'Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch_size by passing a '
                             '`batch_shape` argument to your Input layer.')
        self.reset_buffer()
        super(BufferLayer, self).build(input_shape)

    def call(self, x):
        input_size = x.shape[1].value
        size_diff = self.buffer_size - input_size

        if size_diff > 0:
            #              x -> [[[10], [20]]]
            #    self.buffer -> [[[0], [1], [2]]]
            # updated_buffer -> [[[2], [10], [20]]]
            updated_buffer = K.concatenate(
                tensors=(self.buffer[:, -size_diff:, :], x),
                axis=1
            )
        else:
            #              x -> [[[10], [20], [30], [40]]]
            #    self.buffer -> [[[0], [1], [2]]]
            # updated_buffer -> [[[20], [30], [40]]]
            updated_buffer = x[:, -size_diff:, :]

        self.add_update([(self.buffer, updated_buffer)], x)

        return updated_buffer

    def reset_buffer(self):
        buffer_shape = (self.batch_size, self.buffer_size, self.input_dim)
        if self.buffer is None:
            # initialize to zeros
            self.buffer = K.zeros(buffer_shape)
        else:
            # reset to zeros
            K.set_value(self.buffer, np.zeros(buffer_shape))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.buffer_size, input_shape[-1])
        return output_shape
