
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets('MNIST_DATA_BAK/',one_hot=True)
input_data=np.array(mnist.train.images).reshape(-1,28,28,1)
# print(input_data)

X=tf.placeholder(shape=(None,28,28,1),dtype=tf.float32)
y=tf.placeholder(shape=(None,10),dtype=tf.float32)

W_cnn1=tf.Variable(tf.truncated_normal(shape=(5,5,1,8),stddev=0.1))
b_cnn1=tf.Variable(tf.ones(shape=(8)))
cnn1=tf.nn.conv2d(input=X,filter=W_cnn1,strides=[1,1,1,1],padding='SAME')+b_cnn1
relu1=tf.nn.relu(cnn1)
pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_cnn2=tf.Variable(tf.truncated_normal(shape=(5,5,8,16),stddev=0.1))
b_cnn2=tf.Variable(tf.ones(shape=(16)))
cnn2=tf.nn.conv2d(input=pool1,filter=W_cnn2,strides=[1,1,1,1],padding='SAME')+b_cnn2
relu2=tf.nn.relu(cnn2)
pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

pool2=tf.reshape(pool2,shape=(-1,7*7*16))

w1=tf.Variable(tf.truncated_normal(shape=(7*7*16,100),stddev=0.1),dtype=tf.float32)
b1=tf.Variable(tf.zeros(shape=(100)))
fc1=tf.matmul(pool2,w1)+b1
fc1_relu=tf.nn.relu(fc1)

w2=tf.Variable(tf.truncated_normal(shape=(100,20),stddev=0.1),dtype=tf.float32)
b2=tf.Variable(tf.zeros(shape=(20)))
fc2=tf.matmul(fc1_relu,w2)+b2
fc2_relu=tf.nn.relu(fc2)

w3=tf.Variable(tf.truncated_normal(shape=(20,10),stddev=0.1),dtype=tf.float32)
b3=tf.Variable(tf.zeros(shape=(10)))
sotfmax=tf.matmul(fc2_relu,w3)+b3
y_prediction=tf.nn.softmax(sotfmax)
'''
#注意，这里在计算损失函数的时候，交叉熵又整错了。。。
能不能好好的保证能将交叉熵用对？!
其实也不用纠结，也不需要去记，在使用的时候去查一下，再转换成代码即可。

'''
loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_prediction),reduction_indices=[1]))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss=loss)

'''
在评价这里也是，感觉很迷茫，其实很简单啊
'''
with tf.name_scope('acc'):
    # correct=tf.nn.in_top_k(sotfmax,y,1)
    correct=tf.equal(tf.arg_max(y_prediction,1),tf.arg_max(y,1))
    # acc=tf.reduce_mean(correct)
    acc=tf.reduce_mean(tf.cast(correct,tf.float32))

n_epochs=10000
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    for i in range(n_epochs):
        X_batch,y_batch=mnist.train.next_batch(batch_size=50)
        X_batch=np.array(X_batch).reshape(-1,28,28,1)
        train.run(feed_dict={X:X_batch,y:y_batch})
        if i%100==0:
        #     print('train',acc.eval(feed_dict={X:X_batch,y:y_batch}))
            x_test=np.array(mnist.test.images).reshape(-1,28,28,1)
            y_test=mnist.test.labels
            print('test:',acc.eval(feed_dict={X:x_test,y:y_test}))