import numpy as np

from tensorflow.contrib import legacy_seq2seq

row_zero = np.zeros(shape=(1,10))
row_zero[0, -1] = 1
print(row_zero)

decoder=legacy_seq2seq.rnn_decoder()