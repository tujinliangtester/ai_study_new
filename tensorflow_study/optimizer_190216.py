import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np

n_epochs=50000
learning_rate=0.01
housing_data = fetch_california_housing(data_home='D:\\学习笔记\\ai\\dataSets', download_if_missing=True)

#data  target feature_names

data=housing_data['data']
print(data.shape)
bia=np.float32((np.ones(shape=(data.shape[0],1))))
data_bia=np.float32(np.hstack((bia,data)))

target=housing_data['target']
target=np.float32(target.reshape(-1,1))

#tensoflow
x=tf.constant(data_bia,dtype=tf.float32)
y=tf.constant(target,dtype=tf.float32)

x_mini=tf.placeholder(dtype=tf.float32,shape=(None,9))
y_mini=tf.placeholder(dtype=tf.float32,shape=(None,1))


theta=tf.Variable(tf.ones(shape=(data_bia.shape[1],1)),dtype=tf.float32)
# theta = tf.Variable(tf.random_uniform([data_bia.shape[1], 1], -1.0, 1.0), name='theta')

y_hat=tf.matmul(x,theta)
erro=y-y_hat
loss=tf.reduce_mean(tf.square(erro))


y_hat_mini=tf.matmul(x_mini,theta)
erro_mini=y_mini-y_hat_mini
loss_mini=tf.reduce_mean(tf.square(erro_mini))
'''
用梯度下降会导致梯度爆炸？从而引起损失不减小反而增加？有可能是因为记录数量太大了，导致在计算的过程中数据溢出
用SGD或mini batch GD可能会正常
事实证明，用mini batch还是不行
'''
# optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01,name='GradientDescentOptimizer')
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name='GradientDescentOptimizer')
optimizer_opt=optimizer.minimize(loss)

optm=tf.train.AdamOptimizer(learning_rate=0.01)
opt=optm.minimize(loss_mini)

init_g = tf.global_variables_initializer()
'''

with tf.Session() as sess:
    init_g.run()
    for n_epoch in range(n_epochs):
        optimizer_opt.run()
        learning_rate=learning_rate*learning_rate/(learning_rate+n_epoch)
        if(n_epoch%1000==0):
            # print(erro)
            print(loss.eval())
'''

batch_num=2

with tf.Session() as sess_mini:
    init_g.run()
    next_batch=0
    for n_epoch in range(n_epochs):
        next_batch=next_batch+1
        if(n_epoch+1>data_bia.shape[0]/batch_num):
            next_batch=0
        opt.run(feed_dict={x_mini:data_bia[next_batch*batch_num:(next_batch+1)*batch_num,],
                                     y_mini:target[next_batch*batch_num:(next_batch+1)*batch_num,]})
        # learning_rate=learning_rate*learning_rate/(learning_rate+n_epoch)
        if(n_epoch%10==0):
            # print(erro)
            print(loss_mini.eval(feed_dict={x_mini:data_bia[next_batch*100:(next_batch+1)*100,],
                                     y_mini:target[next_batch*100:(next_batch+1)*100,]}))

