from __future__ import division, print_function

import numpy as np
from keras import Input, Model

from keras_contrib.layers import BufferLayer

np.random.seed(10)


def get_data(n_samples, time_steps, input_dim):
    """Creates and return data of shape (n_samples, time_steps, input_dim).
    Example:
        get_data(n_samples=2, time_steps=3, input_dim=1) ->
        [
          [
            [10.], [11.], [12.]
          ],
          [
            [20.], [21.], [22.]
          ]
        ]

    Returns:
        numpy.ndarray
    """

    def get_sample(_id):
        return (
            (_id + 1) * np.ones((time_steps, input_dim)) * 10 +
            np.array(range(time_steps)).repeat(input_dim).reshape(-1,
                                                                  input_dim)
        )

    return np.array([get_sample(i) for i in range(n_samples)])


time_steps = 3
input_dim = 1
batch_size = 1

buffer_size = 7

input_sequence = Input(batch_shape=(batch_size, time_steps, input_dim))
buffered_sequence = BufferLayer(buffer_size=buffer_size)(input_sequence)
buffer_layer = Model(inputs=input_sequence, outputs=buffered_sequence)
buffer_layer.compile(loss="mse", optimizer='adam')

n_samples = 3
sequences = get_data(n_samples, time_steps, input_dim)
# sequences ->
# [
#     [
#         [10.], [11.], [12.]
#     ],
#     [
#         [20.], [21.], [22.]
#     ],
#     [
#         [30.], [31.], [32.]
#     ]
# ]

output = buffer_layer.predict(sequences, batch_size=batch_size)

# output ->
# [
#     [
#         [0.], [0.], [0.], [0.], [10.], [11.], [12.]
#     ],
#
#     [
#         [0.], [10.], [11.], [12.], [20.], [21.], [22.]
#     ],
#
#     [
#         [12.], [20.], [21.], [22.], [30.], [31.], [32.]
#     ]
#
# ]
