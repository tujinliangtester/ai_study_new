import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.contrib import legacy_seq2seq
from speech_recognition.voip_en import  handle_raw_data,seq2seq_modifyed
from tensorflow.python.ops import variable_scope



# 为了能够使用for循环来迭代Tensor objects
#这里有问题，用了这个就不能用placeholder和session了。。。有时间了还是需要好好研究一下
# tf.enable_eager_execution()

# 一条数据的秒数
n_sec_wav = 8
# 采样率
rate_wav = 16000

# 这是指rnn内核中的单元数量，同时，这也是rnn单元输出的结果维度数量
lstm_num_units_encoder = 100
lstm_num_units_decoder = 100
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
batch_size = 100

data_set_dir='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData'

x = tf.placeholder(shape=(None, diminput), dtype=tf.float32)
y = tf.placeholder(shape=(None, max_line_char_num,n_classes), dtype=tf.float32)

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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_encoder)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return {'LSTM_O': LSTM_O, 'LSTM_S': LSTM_S}


encoder_rnn = RNN(x, num_rnn_layers=num_rnn_layers)
batch_lstm_os = encoder_rnn['LSTM_O']
batch_lstm_o = tf.concat(batch_lstm_os, axis=1)

# 采用loop fun来构建decoder
# 这里采用较简单的loop fun ，即用前一个输出当作后一个输入
def loop_function(prev, _):
    return prev

def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    i=0

    # 这里并没有将state返回，当然目前的这个模型中暂时用不到这个
    def map_fn_fn(inp,prev=prev,state=state,i=i):
        if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
                inp = loop_function(prev, i)
        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
        output, state = cell(inp, state)
        outputs.append(output)
        if loop_function is not None:
            prev = output
        i+=1

    tf.map_fn(fn=lambda inp:map_fn_fn(inp=inp,prev=prev,state=state,i=i)
              ,elems=decoder_inputs)
  return outputs, state

def rnn_decoder_no_iteration(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    # 注意，这里有loop_fun ，不需要再为每一个decoder单元提供输入，则不需要再用迭代循环decoder_inputs了
    # for i, inp in enumerate(decoder_inputs):
    for i in range(max_line_char_num):
        inp=decoder_inputs
        if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
                inp = loop_function(prev, i)
        # if i > 0:
        #     variable_scope.get_variable_scope().reuse_variables()
        output, state = cell(inp, state)
        outputs.append(output)
        if loop_function is not None:
            prev = output
  return outputs, state

# 注意，这里为了能保证在decoder计算时，不会出现维度不匹配的情况，需要先将encoder的输出结果进行一次处理
# 处理方式暂用全连接，名称为 transit 层，但需要留意其中的维度设置
w_transit=tf.Variable(tf.random_normal(
    shape=(lstm_num_units_encoder*num_rnn_layers, lstm_num_units_decoder), dtype=tf.float32))
b_transit=tf.Variable(tf.zeros(shape=(lstm_num_units_decoder)))
transit_out=tf.matmul(batch_lstm_o,w_transit)+b_transit

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_decoder)

decoder_outs, decoder_state = rnn_decoder_no_iteration(
    decoder_inputs=transit_out,
    initial_state=encoder_rnn['LSTM_S'],
    cell=lstm_cell,
    loop_function=loop_function)

sotfmax_outs=[]
weights=[]
for decoder_out in decoder_outs:
    # 这里的softmax其实还是应该用不同的w和b，但可以先共用尝试着训练一下
    sotfmax_out = tf.nn.softmax(tf.matmul(decoder_out, W['w_sotfmax']) + B['b_sotfmax'])
    sotfmax_outs.append(sotfmax_out)
    weights.append(1)
# 此weights是代表，各个sotfmax_out之间的loss占总loss的权重，暂设为1
# loss =  seq2seq_modifyed.sequence_loss(targets=y, logits=sotfmax_outs,weights=weights)

loss=0
for i,sotfmax_out in enumerate(sotfmax_outs):
    loss_tmp=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y[:,i,:],logits=sotfmax_out))
    # 注意，如果这里用loss相乘的话，得到的模型，计算出acc会非常低，原因是什么呢？
    # 有可能是因为这里的每一个decoder rnn不是通用的Varilble，相乘肯定会出问题？
    loss+=loss_tmp

optm = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = optm.minimize(loss=loss)

correct_prediction=0
for i,logit in enumerate(sotfmax_outs):
    target=y[:,i,:]
    correct_prediction += tf.cast(tf.equal(tf.cast(tf.argmax(logit, 1), tf.float32), tf.cast(tf.argmax(target, 1), tf.float32)),tf.float32)

correct_prediction=correct_prediction/max_line_char_num
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ini = tf.global_variables_initializer()

with tf.Session() as sess:
    fw = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
    saver = tf.train.Saver()
    ini.run()
    n_iter = 0
    for i in range(n_epoches):
        indexs=[]
        if (n_iter + batch_size > 1000):
            n_iter = 0
        for j in range(n_iter, n_iter + batch_size):
            indexs.append(j)
        y_=handle_raw_data.get_y(indexs=indexs,datapath=data_set_dir,n=max_line_char_num)
        opt.run(feed_dict={x: x_[n_iter:n_iter + batch_size], y: y_})
        print('i', i, 'train acc', acc.eval(feed_dict={x: x_[n_iter:n_iter + batch_size], y: y_}))
        saver.save(sess, save_path='save_sess/')
        if (i % 100 == 0):
            print('i', i, 'train acc', acc.eval(feed_dict={x: x_[n_iter:n_iter + batch_size], y: y_}))
            saver.save(sess, save_path='save_sess/')
        n_iter = n_iter + batch_size
