
import numpy as np
import tensorflow as tf

num_units=5
number_of_layers=10
batch_size=200
num_steps=[1]
words=np.zeros(shape=(2,2,2))


lstm=tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * number_of_layers)
initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(len(num_steps)):
    # 每次处理一批词语后更新状态值.
    output, state = lstm(words[:,i], state)

    # 其余的代码.
    # ...

final_state = state

with tf.Session() as sess:
    fw=tf.summary.FileWriter(logdir='/logs',graph=sess.graph)