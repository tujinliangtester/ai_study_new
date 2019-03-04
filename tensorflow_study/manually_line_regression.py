from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_epochs = 2
learning_rate = 0.001
housing_data = fetch_california_housing(data_home='D:\\学习笔记\\ai\\dataSets', download_if_missing=True)
data = housing_data.data
print(housing_data.feature_names)
m, n = data.shape
# 这一步numpy是直接执行的，不能放到tf里面去执行吗？
housing_data_puls_bias = np.c_[np.ones((m, 1)), data]

X = tf.constant(housing_data_puls_bias, name='X', dtype=tf.float32)
y = tf.constant(housing_data.target.reshape(-1, 1), name='y', dtype=tf.float32)
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
print('theta:', theta.shape)
y_hat = tf.matmul(X, theta, name='y_hat')
err = y_hat - y
print(y_hat.shape, y.shape)
print('err.shape:', err.shape)

mse = tf.reduce_mean(tf.square(err), name='mse')
gradient = 2 / m * tf.matmul(tf.transpose(X), err)
gradient2=tf.gradients(mse,[theta])[0]
#下面这一行可能有问题！ 迭代这个地方与教材不一样，算出来的提跌替换是错误的！可能是由于TensorFlow版本不一样导致
theta_1=theta - tf.Variable(learning_rate * gradient)
theta_1=theta-learning_rate*gradient
train_op = tf.assign(theta, theta-learning_rate*gradient)

init = tf.global_variables_initializer()
print(X.shape)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        if epoch%100==0:
            print('theta', theta.eval())
            print('theta_1', theta_1.eval())
            print('gradient:',gradient.eval())
            # print('mse.eval():',mse.eval())
            # # print(gradient.eval())
            # print('err:',err.eval())
            print('gradient2', gradient2.eval())
            train_op.eval()
    print('theta:', theta.eval())
'''
问题，为什么mse越来越大？
通过打印发现，err越来越大，那么一定是计算y_hat或theta迭代异常
计算y_hat应该没有什么问题，关键就在迭代上

'''