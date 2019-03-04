import tensorflow as tf
from cifar10 import cifar10_input,cifar10

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size=100
# data_dir='D:\\学习笔记\\ai\\dataSets\\cifar10\\cifar-10-binary.tar'
data_dir='D:\\学习笔记\\ai\\dataSets\\cifar10\\cifar-10-batches-bin'
# data_dir='D:\\学习笔记\\ai\\dataSets\\cifar10'

cifar10.maybe_download_and_extract()

images_train,label_train= cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test,label_test= cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

X=tf.placeholder(shape=images_train.get_shape(),dtype=tf.float32)
y=tf.placeholder(shape=label_train.get_shape(),dtype=tf.int32)
w_cnn=tf.Variable(tf.truncated_normal(shape=(4,4,3,24),stddev=0.1))
b_cnn=tf.Variable(tf.ones(shape=24))
cnn=tf.nn.conv2d(X,w_cnn,strides=(1,1,1,1),padding='SAME',name='cnn')
cnn=tf.reshape(cnn,shape=(batch_size,-1))
cnn_shape=cnn.get_shape()
relu_cnn=tf.nn.relu(cnn)

print('cnn shape',cnn_shape)
w_fc=tf.Variable(tf.truncated_normal(shape=(13824,10),stddev=0.1))
b_fc=tf.Variable(tf.ones(shape=10))
fc=tf.matmul(relu_cnn,w_fc)+b_fc

# print('fc shape :',fc.get_shape())
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc,labels=y,name='loss')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001,name='optimizer')
train=optimizer.minimize(loss=loss)

acc=tf.nn.in_top_k(fc,y,1)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    images_train_batch,label_train_batch=sess.run([images_train,label_train])
    print(2222222222222)
    print(images_train_batch)
    train.run(feed_dict={X:images_train_batch,y:label_train_batch})
    print('acc:')
    print(acc.eval())