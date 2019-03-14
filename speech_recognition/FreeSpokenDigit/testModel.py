import tensorflow as tf
from tensorflow.contrib import rnn
from speech_recognition.FreeSpokenDigit import LoadData
from sklearn.metrics import accuracy_score
import numpy as np

# 一条数据的秒数
n_sec_wav = 3
# 采样率
rate_wav = 8000

# 这是指rnn内核中的单元数量，同时，这也是rnn单元输出的结果维度数量
lstm_num_units_encoder = 100
lstm_num_units_decoder = 10
# 分类的种类数量
n_classes = 10
# 一个音频文件的大小
diminput = n_sec_wav * rate_wav
# 注意，这不是一个音频文件被切分成多少份，而是指样本总量分成多少批次进行训练，
# 由于机器性能原因，这里可以设置大一点，训练久一些

# 这与需要将一条数据切分成多少小节有关，一般认为20ms长度的数据是比较合适的
# 所以，num_rnn_layers=一条数据秒数*1000/20
num_rnn_layers = int(n_sec_wav * 1000 / 20)

learning_rate = 0.001

# 训练循环次数
n_epoches = 10000
batch_size=100

x = tf.placeholder(shape=(None, diminput), dtype=tf.float32)
y = tf.placeholder(shape=(None, n_classes), dtype=tf.float32)

x_, y_ = LoadData.load_data(one_hot=True,filepath='D:\\学习笔记\\ai\\dataSets\\number-wav-data\\test\\')

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
    '''
    # 注意，这里rnn.static_rnn这个方法，inputs=x，是将x的每一行带入到lstm_cell中进行计算，
    # 并将结果保存到(outputs, state)
    # 所以，其实就相当于是说，行数代表了rnn的层数；
    # outputs是每一层的output组合成的列表；
    # state只会保存最后一个state
    并且这里需要特别注意的是：
        这里并没有loop_fun，所以，rnn相互之间只有一个state进行了关联，输入输出之间并没有直接进行影响
        如果直接用最后一个输出的结果，很显然是不合理的，需要进行调整
            方法一：将每一个输出进行concat，即合并，然后再作为decoder的输入
            方法二：不用rnn.static_rnn，改用其他方法，使得encoder的每一个rnn相互之间输出更换输入
                （但是原始输入又没法用了，所以如果要用这种方法的话，还需要对loop_fun进行整理，可能会更加复杂）
    '''
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x,dtype=tf.float32)

    return {'LSTM_O': LSTM_O, 'LSTM_S': LSTM_S}

'''
其实这个方法没有用到

def batch_inputs_rnn(inputs):
    batch_lstm_o=np.mat(np.zeros(shape=(1,lstm_num_units_encoder)))
    # map_fn代替for循环
    # myrnn=tf.map_fn(fn=lambda input:RNN(input,num_rnn_layers=num_rnn_layers),elems=inputs)
    myrnn=RNN(inputs,num_rnn_layers=num_rnn_layers)
    batch_lstm_o=myrnn['LSTM_O']
    # for input in inputs:
    #     myrnn = RNN(input, num_rnn_layers=num_rnn_layers)
    #     #取最后一行，即最后一个output，并加到batch_lstm_o中
    #     last_LSTM_O=myrnn['LSTM_O'][-1]
    #     batch_lstm_o = np.vstack((batch_lstm_o, last_LSTM_O))
    return batch_lstm_o[-1]
'''

batch_lstm_os=RNN(x,num_rnn_layers=num_rnn_layers)['LSTM_O']
batch_lstm_o=tf.concat(batch_lstm_os,axis=1)


# decoder  目前只有一层，所以直接用全连接+softmax，后续可以如法套用lstm
decoder_out = tf.matmul(batch_lstm_o, W['w_decoder']) + B['b_decoder']

sotfmax_out = tf.matmul(decoder_out, W['w_sotfmax']) + B['b_sotfmax']

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=sotfmax_out)
optm = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = optm.minimize(loss=loss)

# acc = accuracy_score(y_true=y, y_pred=sotfmax_out)#事实证明，用sklearn的acc不行
pre=tf.argmax(sotfmax_out, 1)
correct_prediction = tf.equal(tf.cast(tf.argmax(sotfmax_out, 1), tf.float32), tf.cast(tf.argmax(y, 1),tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ini = tf.global_variables_initializer()

with tf.Session() as sess:
    # fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    saver=tf.train.Saver()
    ini.run()
    '''
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
    1_tujinliang_0
    '''
    saver.restore(sess,save_path='save_sess/')
    print('total acc', acc.eval(feed_dict={x: x_, y: y_}))
    print('pre',pre.eval(feed_dict={x: x_, y: y_}))