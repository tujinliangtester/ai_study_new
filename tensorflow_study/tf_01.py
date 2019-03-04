import tensorflow as tf
import numpy as np
from numpy import random
# x=tf.constant(random.rand(100,3))
x=random.rand(100,3)
theta0=[[1],[2],[3]]
y=np.matmul(x,theta0)

x=tf.constant(x,dtype=tf.float32)
y=tf.constant(y,dtype=tf.float32)
theta=tf.Variable(tf.ones(shape=(3,1)))


y_hat=tf.matmul(x,theta)
error=y-y_hat
mse=tf.reduce_mean(tf.square(error))

opt=tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt_opt=opt.minimize(mse)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(10000):
        opt_opt.run()
        print(mse.eval())
    print(theta.eval())


