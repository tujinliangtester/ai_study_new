import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.contrib import legacy_seq2seq
from speech_recognition.voip_en import  handle_raw_data

# 为了能够使用for循环来迭代Tensor objects
#这里有问题，用了这个就不能用placeholder和session了。。。有时间了还是需要好好研究一下
tf.enable_eager_execution()

# 一条数据的秒数
n_sec_wav = 8
# 采样率
rate_wav = 16000

# 这是指rnn内核中的单元数量，同时，这也是rnn单元输出的结果维度数量
lstm_num_units_encoder = 10
lstm_num_units_decoder = 10
# 分类的种类数量，前1000个音频文件，总共有186个单词，加一个空单词
n_classes = 187

# 每个音频文件的单词数量，在前1000条音频文件中，最大的有18个单词，暂取20个
max_line_char_num=20

# 一个音频文件的大小
diminput = n_sec_wav * rate_wav

# 这与需要将一条数据切分成多少小节有关，一般认为20ms长度的数据是比较合适的
# 所以，num_rnn_layers=一条数据秒数*1000/20
num_rnn_layers = int(n_sec_wav * 1000 / 20)

learning_rate = 0.001

# 训练循环次数
n_epoches = 10000

# 所以考虑到内存爆炸，暂定5个文件为一个批次
batch_size = 5

data_set_dir='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData'

x_=handle_raw_data.get_x(x_path='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\x1.npy')

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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_encoder,dtype=tf.float32)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return {'LSTM_O': LSTM_O, 'LSTM_S': LSTM_S}




# 采用loop fun来构建decoder
# 这里采用较简单的loop fun ，即用前一个输出当作后一个输入
def loop_function(prev, _):
    return prev

# ini = tf.global_variables_initializer()

# fw = tf.summary.FileWriter(logdir='logs/')
# saver = tf.train.Saver()
# ini.run()
n_iter = 0

for i in range(n_epoches):
    indexs = []
    if (n_iter + batch_size > 1000):
        n_iter = 0
    for i in range(n_iter, n_iter + batch_size):
        indexs.append(i)
    x=x_[n_iter:n_iter + batch_size]
    y=handle_raw_data.get_y(indexs=indexs, datapath=data_set_dir, n=max_line_char_num)

    encoder_rnn = RNN(x, num_rnn_layers=num_rnn_layers)
    batch_lstm_os = encoder_rnn['LSTM_O']
    batch_lstm_o = tf.concat(batch_lstm_os, axis=1)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_decoder)
    decoder_outs, decoder_state = legacy_seq2seq.rnn_decoder(
        decoder_inputs=batch_lstm_o,
        initial_state=encoder_rnn['LSTM_S'],
        cell=lstm_cell,
        loop_function=loop_function)

    sotfmax_outs = []
    weights = []
    for decoder_out in decoder_outs:
        # 这里的softmax其实还是应该用不同的w和b，但可以先共用尝试着训练一下
        sotfmax_out = tf.matmul(decoder_out, W['w_sotfmax']) + B['b_sotfmax']
        sotfmax_outs.append(sotfmax_out)
        weights.append(1)
    # 此weights是代表，各个sotfmax_out之间的loss占总loss的权重，暂设为1
    loss = legacy_seq2seq.sequence_loss_by_example(targets=y, logits=sotfmax_outs, weights=weights)

    optm = tf.train.AdamOptimizer(learning_rate=learning_rate)
    opt = optm.minimize(loss=loss)

    correct_prediction = 0
    for logit, target in zip(sotfmax_outs, y):
        correct_prediction += tf.equal(tf.cast(tf.argmax(logit, 1), tf.float32),
                                       tf.cast(tf.argmax(target, 1), tf.float32))

    correct_prediction = correct_prediction / max_line_char_num
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if (i % 100 == 0):
        print('i', i, 'train acc', acc)
    n_iter = n_iter + batch_size



'''
with tf.Session() as sess:
    fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    saver = tf.train.Saver()
    ini.run()
    n_iter = 0
    for i in range(n_epoches):
        indexs=[]
        if (n_iter + batch_size > 1000):
            n_iter = 0
        for i in range(n_iter, n_iter + batch_size):
            indexs.append(i)
        y_=handle_raw_data.get_y(indexs=indexs,datapath=data_set_dir,n=max_line_char_num)
        opt.run(feed_dict={x: x_[n_iter:n_iter + batch_size], y: y_})
        if (i % 100 == 0):
            print('i', i, 'train acc', acc.eval(feed_dict={x: x_[n_iter:n_iter + batch_size], y: y_}))
            saver.save(sess, save_path='save_sess/')
        n_iter = n_iter + batch_size
'''