from __future__ import division, print_function

import numpy as np
from keras import Input, Model

from keras_contrib.layers import BufferLayer

np.random.seed(10)


def get_data(n_samples, time_steps, input_dim):
    return np.random.rand(n_samples, time_steps, input_dim)


def get_structured_data(n_samples, time_steps, input_dim):
    def get_sample(_id):
        return (
            (_id + 1) * np.ones((time_steps, input_dim)) * 10 +
            np.array(range(time_steps)).repeat(input_dim).reshape(-1, input_dim)
        )

    return np.array([get_sample(i) for i in range(n_samples)])


n_samples = 10
time_steps = 5
input_dim = 3
batch_size = 2

input_ = Input(batch_shape=(batch_size, time_steps, input_dim))
buffer = BufferLayer(buffer_size=7, input_dim=input_dim)
print('initialized')
buffered = buffer(input_)
model = Model(inputs=input_, outputs=buffered)

model.compile(loss="mse", optimizer='adam')
data = get_structured_data(n_samples, time_steps, input_dim)
output = model.predict(data, batch_size=batch_size)
