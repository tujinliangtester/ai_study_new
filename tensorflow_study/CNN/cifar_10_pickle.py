
'''



'''
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
n_epoch = 0
batch_size = 100
data_dir = 'D:\\学习笔记\\ai\\dataSets\\cifar-10-batches-py\\'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_dir_data_batch_1 = data_dir + 'data_batch_1'
train_dict = unpickle(data_dir_data_batch_1)
images_train, label_train = train_dict[b'data'], train_dict[b'labels']
images_train = np.array(images_train).reshape(-1, 32, 32, 3)
# images_train = np.array(images_train)

data_dir_test = data_dir + 'test_batch'
test_dict = unpickle(data_dir_test)
images_test, label_test = test_dict[b'data'], test_dict[b'labels']
# images_test = np.array(images_test).reshape(-1, 32, 32, 3)
# images_train,label_train= cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# images_test,label_test= cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

X = tf.placeholder(shape=(batch_size, 32, 32, 3), dtype=tf.float32)
# X = tf.placeholder(shape=(batch_size,images_train.shape[1]), dtype=tf.float32)
# X=tf.reshape(X,shape=(batch_size,32,32,3))
# y = tf.placeholder(shape=(batch_size), dtype=tf.float32)
y = tf.placeholder(shape=(batch_size), dtype=tf.float32)

w_cnn = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 24), stddev=0.01))
b_cnn = tf.Variable(tf.zeros(shape=24))
cnn = tf.nn.conv2d(X, w_cnn, strides=(1, 1, 1, 1), padding='SAME', name='cnn') + b_cnn
relu_cnn = tf.nn.relu(cnn)
pool_cnn=tf.nn.max_pool(relu_cnn,ksize=(1,2,2,1),padding='SAME',strides=(1,2,2,1))
pool_cnn=tf.reshape(pool_cnn,shape=(batch_size,-1))
'''

w_cnn2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 24, 48), stddev=0.01))
b_cnn2 = tf.Variable(tf.ones(shape=48))
cnn2 = tf.nn.conv2d(pool_cnn, w_cnn2, strides=(1, 1, 1, 1), padding='SAME', name='cnn') + b_cnn2
relu_cnn2 = tf.nn.relu(cnn2)
pool_cnn2=tf.nn.max_pool(relu_cnn2,ksize=(1,2,2,1),padding='SAME',strides=(1,2,2,1))
pool_cnn2=tf.reshape(pool_cnn2,shape=(batch_size,-1))
'''
w_fc = tf.Variable(tf.truncated_normal(shape=(tf.shape(pool_cnn)[1], 10), stddev=0.1))
b_fc = tf.Variable(tf.zeros(shape=10))
fc = tf.matmul(pool_cnn, w_fc) + b_fc

#2.3033
'''
relu_fc = tf.nn.relu(fc)
w_fc2 = tf.Variable(tf.truncated_normal(shape=(tf.shape(fc)[1], 10), stddev=0.1))
b_fc2 = tf.Variable(tf.zeros(shape=10))
fc2 = tf.matmul(relu_fc, w_fc2) + b_fc2
'''

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc, labels=tf.cast(y, tf.int32), name='loss')
loss=tf.reduce_mean(loss)

def learning_rate_min(learning_rate, n_epoch):
    n_epoch = n_epoch + 1
    return learning_rate / (learning_rate + n_epoch), n_epoch


learning_rate, n_epoch = learning_rate_min(0.1, n_epoch)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001, name='optimizer')
train = optimizer.minimize(loss=loss)

# correct = tf.nn.in_top_k(fc, tf.cast(y, tf.int32), 1)
# correct = tf.reduce_sum(tf.cast(correct, tf.float32))
# acc = tf.reduce_sum(correct) / batch_size
init = tf.global_variables_initializer()
with tf.Session() as sess:
    count=0
    j=-1
    n = 1
    writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    init.run()
    for i in range(100000):
        j+=1
        n+=1
        if(j>(10000-batch_size)/batch_size):
            # j=int(j%((10000-batch_size)/batch_size))
            j=0
            # count=0
            n=1
        # print(j)
        images_train_batch, label_train_batch = images_train[j*batch_size:batch_size*(j+1), :,:,:], \
                                                label_train[j*batch_size:batch_size*(j+1)]
        # images_train_batch, label_train_batch = images_train[batch_size:2*batch_size, :, :, :], \
        #                                         label_train[batch_size:2*batch_size]
        # images_train_batch, label_train_batch = images_train[batch_size:2*batch_size,:], \
        #                                         label_train[batch_size:2 * batch_size]

        _,loss_train=sess.run([train,loss],feed_dict={X: images_train_batch, y: label_train_batch})
        if (i % 100 == 0):
            print(i,'acc:','loss:',loss_train)
