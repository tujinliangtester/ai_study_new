
'''
思想很重要，算法是实现思想，得出结果，实现必须执行
tf：
1.首先画出计算图
2.图的每一个节点构建，可能是constant，可能是varible，可能是placeholder
3.在图中使用解决思想的优化方法，进行迭代
4.真正在session中,给数据并进行计算，得出结果
5.评估结果，当结果相差甚远的时候，有两种思路，从根往叶找，或从叶往根找。在此之前，首先应将优化算法相关的东西全部砍掉！
    所以，步骤应为：
        1.逐步砍掉优化算法
        2.模型是否正确
        3.构建图是否正确
        4.数据是否正确
        5.计算是否正确
        6.评估是否正确
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
import numpy as np

n_features=28*28
n_hidden1=300
n_hidden2=100
n_output=10

with tf.name_scope('para'):
    X=tf.placeholder(tf.float32,shape=(None,n_features),name='features')

y=tf.placeholder(tf.int64,shape=(None),name='y')

def neuron_layer(input,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_input=int(input.get_shape()[1])
        stddev = 2 / np.sqrt(n_input)
        # stddev=0.1
        initw = tf.random_normal((n_input, n_neurons), stddev=stddev)  # y_=x.w+b
        w=tf.Variable(initw,name='weight')
        tf.summary.histogram('weight',w)
        b = tf.Variable(tf.zeros([n_neurons]),name='biases')
        tf.summary.histogram('b',b)
        z=tf.matmul(input,w)+b
        if (activation=='relu'):
            return tf.nn.relu(z),w,b
        else:
            return z,w,b

with tf.name_scope('dnn'):
    hidden1,w1,b1=neuron_layer(X,n_hidden1,'hidden1',activation='relu')
    hidden2,w2,b2=neuron_layer(hidden1,n_hidden2,'hidden2',activation='relu')
    logits,w3,b3=neuron_layer(hidden2,n_output,'softmax')
'''
with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    logits = fully_connected(hidden2, n_output, scope='softmax',activation_fn=None)
'''

with tf.name_scope('loss'):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name='loss')
    tf.summary.scalar('loss',loss)

with tf.name_scope('optimizer'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train=optimizer.minimize(loss)#这里会对tf图中的varible节点(这里指的是w和w0）进行梯度下降

with tf.name_scope('eval'):
    correct=tf.nn.in_top_k(logits,y,1)
    acc=tf.reduce_mean(tf.cast(correct,tf.float32))

saver=tf.train.Saver()

mnist=input_data.read_data_sets('MNIST_DATA_BAK/')
batch_size=50
mini_batch_epoch=int(mnist.train.num_examples/batch_size)
n_epochs=10
with tf.Session() as sess:
    merged=tf.summary.merge_all()#将各类型的结果保存merge到一起，在一个网页中查看
    # sess.run(merged)
    writer = tf.summary.FileWriter('logs/', sess.graph)

    init = tf.global_variables_initializer()
    init.run()
    for epoch in range(n_epochs):
        for i in range(mini_batch_epoch):
            X_batch,y_batch=mnist.train.next_batch(batch_size=batch_size)
            train.run(feed_dict={X:X_batch,y:y_batch})
            # if(i%50==0):
                # acc_train=acc.eval(feed_dict={X:X_batch,y:y_batch})
                # print('w:',w1.eval())
                # print(i,acc_train)
                #问题，为什么正确率这么低？！
                #事实证明，自己构造的图是错误的。
                #1208  考虑两个问题：1.为什么board有时候显示不出来？2.研究画出的graph，找出问题究竟出在哪里
                #1211  事实证明，无论是从代码流程上，还是从tensorboard画的图来看，表现的层级构建图与初始设计是一样的。
                #而问题却出现在stddev上！！！说明，在最开始做的时候，不要盲目的加一些优化的算法，否则可能会导致出现问题。

        acc_test=acc.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
        print('test:',acc_test)
    # saver.save(sess,'./')