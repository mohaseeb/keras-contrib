from keras.engine import Layer
from keras.engine import InputSpec
from .. import backend as K

import numpy as np


class BufferLayer(Layer):

    def __init__(self, buffer_size, input_dim, **kwargs):
        self.buffer_size = buffer_size
        self.input_dim = input_dim
        self.batch_size = None
        self.buffer = None
        super(BufferLayer, self).__init__(**kwargs)
        self.stateful = True

    def build(self, input_shape):
        self.batch_size = input_shape[0]
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
        print('built')

    def call(self, x):
        print('call')
        input_size = x.shape[1].value
        keep_size = self.buffer_size - input_size

        if keep_size > 0:
            old = self.buffer[:, :keep_size, :]
            old =  K.print_tensor(old, message='y_true = ')
            # old = K.ones((self.batch_size, keep_size, self.input_dim))
            new_buffer = K.concatenate((x, old), axis=1)
        else:
            pass
        self.add_update([(self.buffer, new_buffer)], x)

        return new_buffer

    def reset_buffer(self, buffer=None):
        buffer_shape = (self.batch_size, self.buffer_size, self.input_dim)

        if self.buffer is None:
            # initialize to zeros
            self.buffer = K.zeros(buffer_shape)
        elif buffer is None:
            # reset to zeros
            K.set_value(self.buffer, np.zeros(buffer_shape))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.buffer_size, input_shape[-1])
        return output_shape
