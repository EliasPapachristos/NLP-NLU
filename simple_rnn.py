from __future__ import print_function, division
from builtins import range, input

# I may need to update this version in the future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU

# Dummy stuff
T = 8
D = 2
M = 3


X = np.random.randn(1, T, D)


def lstm_1():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)


def lstm_2():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True, return_sequences=True)
  # rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)


def gru_1():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)


def gru_2():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True, return_sequences=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)



print("lstm_1:")
lstm_1()
print("lstm_2:")
lstm_2()
print("gru_1:")
gru_1()
print("gru_2:")
gru_2()
