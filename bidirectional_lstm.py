from __future__ import print_function, division
from builtins import range, input

# I may need to update this version in the future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU


T = 8
D = 2
M = 3


X = np.random.randn(1, T, D)


input_ = Input(shape=(T, D))
rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
# The result is h1 is the same with the first 3 of the last one
# -0.08713648  0.25286463  0.0664605 and
# h2 is the last 3 of the first one
# -0.03439435  0.0328106  -0.1275148

# Now, comment the above rnn and run this rnn
# rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))
# The result is a concatenation of both h1 & h2 in one row
# -0.11160935  0.07912716 -0.19062757 -0.03882445 -0.03457037 -0.10122014
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
o, h1, c1, h2, c2 = model.predict(X)
print("o:", o)
print("o.shape:", o.shape)
print("h1:", h1)
print("c1:", c1)
print("h2:", h2)
print("c2:", c2)