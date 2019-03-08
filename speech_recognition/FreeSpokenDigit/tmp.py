import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
x=np.ones(shape=(1,20))
x_split=tf.split(x,num_or_size_splits=4,axis=1)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=2)
LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x_split,dtype=tf.float64)
output0=LSTM_O[0]
output1=LSTM_O[1]

ini=tf.global_variables_initializer()
with tf.Session() as sess:
    ini.run()
    saver=tf.train.Saver()
    fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    print(output0.eval())
    print(output1.eval())
    saver.save(sess,save_path='tmp/model.ckpt')
