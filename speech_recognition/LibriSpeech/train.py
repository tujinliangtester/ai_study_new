import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from . import read_xy
from tensorflow.contrib import legacy_seq2seq

# 一条数据的秒数
n_sec_wav = 30
# 采样率
rate_wav = 16000

# 这是指rnn内核中的单元数量，同时，这也是rnn单元输出的结果维度数量
lstm_num_units_encoder = 100
lstm_num_units_decoder = 10
# 分类的种类数量
n_classes = 5575

# 每个音频文件的单词数量，数据中最大是52，暂定60个单词
max_line_char_num=60

# 一个音频文件的大小
diminput = n_sec_wav * rate_wav

# 这与需要将一条数据切分成多少小节有关，一般认为20ms长度的数据是比较合适的
# 所以，num_rnn_layers=一条数据秒数*1000/20
num_rnn_layers = int(n_sec_wav * 1000 / 20)

learning_rate = 0.001

# 训练循环次数
n_epoches = 10000

# 由于一条音频文件就接近30秒，所以考虑到内存爆炸，暂定5个文件为一个批次
batch_size = 5

x = tf.placeholder(shape=(None, diminput), dtype=tf.float32)
y = tf.placeholder(shape=(None, max_line_char_num,n_classes), dtype=tf.float32)

fname_list_flac = read_xy.fname_list_flac
x_ = read_xy.read_flac2x(n_sec_wav=n_sec_wav, rate_wav=rate_wav, batch_fname_list_flac=fname_list_flac[:5])
y_ = read_xy.read_trans2y()

W = {'w_decoder': tf.Variable(
    tf.random_normal(shape=(lstm_num_units_encoder * num_rnn_layers, lstm_num_units_decoder), dtype=tf.float32)),
    'w_sotfmax': tf.Variable(tf.random_normal(shape=(lstm_num_units_decoder, n_classes), dtype=tf.float32))}
B = {'b_decoder': tf.Variable(tf.zeros(shape=(lstm_num_units_decoder))),
     'b_sotfmax': tf.Variable(tf.zeros(shape=(n_classes)))}

# 核心代码
'''
注意，传入这个函数的x，只能是一个批量的音频文件
'''
def RNN(x, num_rnn_layers=num_rnn_layers):
    x = tf.split(x, num_rnn_layers, axis=1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_encoder)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return {'LSTM_O': LSTM_O, 'LSTM_S': LSTM_S}


encoder_rnn = RNN(x, num_rnn_layers=num_rnn_layers)
batch_lstm_os = encoder_rnn['LSTM_O']
batch_lstm_o = tf.concat(batch_lstm_os, axis=1)

# decoder  目前只有一层，所以直接用全连接+softmax，后续可以如法套用lstm
# decoder_out = tf.matmul(batch_lstm_o, W['w_decoder']) + B['b_decoder']

# 采用loop fun来构建decoder
# 这里采用较简单的loop fun ，即用前一个输出当作后一个输入
def loop_function(prev, _):
    return prev

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_decoder)
decoder_outs, decoder_state = legacy_seq2seq.rnn_decoder(
    decoder_inputs=batch_lstm_o,
    initial_state=encoder_rnn['LSTM_S'],
    cell=lstm_cell,
    loop_function=loop_function)

# TODO
# 这里既然能用loop fun 那么，就可以采用类似的手段，将encoder decoder合并在一起，后续todo
sotfmax_outs=[]
weights=[]
for decoder_out in decoder_outs:
    sotfmax_out = tf.matmul(decoder_out, W['w_sotfmax']) + B['b_sotfmax']
    sotfmax_outs.append(sotfmax_out)
    weights.append(1)
# 此weights是代表，各个sotfmax_out之间的loss占总loss的权重，暂设为1
loss =  legacy_seq2seq.sequence_loss_by_example(targets=y, logits=sotfmax_outs,weights=weights)

optm = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = optm.minimize(loss=loss)

# acc = accuracy_score(y_true=y, y_pred=sotfmax_out)#事实证明，用sklearn的acc不行
correct_prediction=0
for logit, target in zip(sotfmax_outs, y):
    correct_prediction += tf.equal(tf.cast(tf.argmax(logit, 1), tf.float32), tf.cast(tf.argmax(y, 1), tf.float32))

correct_prediction=correct_prediction/max_line_char_num
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ini = tf.global_variables_initializer()

with tf.Session() as sess:
    fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    saver = tf.train.Saver()
    ini.run()
    n = 0
    for i in range(n_epoches):
        # todo
        # 需要构造y，感觉又想放弃了呢  方向好像是错的

        if (n + batch_size > 1900):
            n = 0
        opt.run(feed_dict={x: x_[n:n + batch_size], y: y_[n:n + batch_size]})
        if (i % 100 == 0):
            print('i', i, 'train acc', acc.eval(feed_dict={x: x_[n:n + batch_size], y: y_[n:n + batch_size]}))
            saver.save(sess, save_path='save_sess/')
        n = n + batch_size
    print('total acc', acc.eval(feed_dict={x: x_, y: y_}))
