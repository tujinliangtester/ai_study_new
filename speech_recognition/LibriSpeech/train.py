import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from . import read_xy

# 一条数据的秒数
n_sec_wav = 30
# 采样率
rate_wav = 16000

# 这是指rnn内核中的单元数量，同时，这也是rnn单元输出的结果维度数量
lstm_num_units_encoder = 100
lstm_num_units_decoder = 10
# 分类的种类数量
n_classes = 8333
# 一个音频文件的大小
diminput = n_sec_wav * rate_wav

# 这与需要将一条数据切分成多少小节有关，一般认为20ms长度的数据是比较合适的
# 所以，num_rnn_layers=一条数据秒数*1000/20
num_rnn_layers = int(n_sec_wav * 1000 / 20)

learning_rate = 0.001

# 训练循环次数
n_epoches = 10000

#由于一条音频文件就接近30秒，所以考虑到内存爆炸，暂定5个文件为一个批次
batch_size=5

x = tf.placeholder(shape=(None, diminput), dtype=tf.float32)
y = tf.placeholder(shape=(None, n_classes), dtype=tf.float32)

fname_list_flac=read_xy.fname_list_flac
x_=read_xy.read_flac2x(n_sec_wav=n_sec_wav, rate_wav=rate_wav, batch_fname_list_flac=fname_list_flac[:5])
y_ =read_xy.read_trans2y()

W = {'w_decoder': tf.Variable(
    tf.random_normal(shape=(lstm_num_units_encoder*num_rnn_layers, lstm_num_units_decoder), dtype=tf.float32)),
     'w_sotfmax': tf.Variable(tf.random_normal(shape=(lstm_num_units_decoder, n_classes), dtype=tf.float32))}
B = {'b_decoder': tf.Variable(tf.zeros(shape=(lstm_num_units_decoder))),
     'b_sotfmax': tf.Variable(tf.zeros(shape=(n_classes)))}

# 核心代码
'''
注意，传入这个函数的x，只能是一个批量的音频文件
'''
def RNN(x, num_rnn_layers=num_rnn_layers):
    x = tf.split(x, num_rnn_layers,axis=1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_encoder)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x,dtype=tf.float32)
    return {'LSTM_O': LSTM_O, 'LSTM_S': LSTM_S}

batch_lstm_os=RNN(x,num_rnn_layers=num_rnn_layers)['LSTM_O']
batch_lstm_o=tf.concat(batch_lstm_os,axis=1)


# decoder  目前只有一层，所以直接用全连接+softmax，后续可以如法套用lstm
decoder_out = tf.matmul(batch_lstm_o, W['w_decoder']) + B['b_decoder']

sotfmax_out = tf.matmul(decoder_out, W['w_sotfmax']) + B['b_sotfmax']

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=sotfmax_out)
optm = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = optm.minimize(loss=loss)

# acc = accuracy_score(y_true=y, y_pred=sotfmax_out)#事实证明，用sklearn的acc不行
correct_prediction = tf.equal(tf.cast(tf.argmax(sotfmax_out, 1), tf.float32), tf.cast(tf.argmax(y, 1),tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ini = tf.global_variables_initializer()

with tf.Session() as sess:
    fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    saver=tf.train.Saver()
    ini.run()
    n=0
    for i in range(n_epoches):
        if(n+batch_size>1900):
            n=0
        opt.run(feed_dict={x: x_[n:n+batch_size], y: y_[n:n+batch_size]})
        if (i % 100 == 0):
            print('i',i,'train acc',acc.eval(feed_dict={x: x_[n:n+batch_size], y: y_[n:n+batch_size]}))
            saver.save(sess,save_path='save_sess/')
        n=n+batch_size
    print('total acc', acc.eval(feed_dict={x: x_, y: y_}))