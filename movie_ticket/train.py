

import tensorflow as tf
from  movie_ticket import  tmp


n_epoches=100

x_=tmp.read_data('x.npy')
y_=tmp.read_data('y_onehot.npy')

x=tf.placeholder(shape=(None,2),dtype=tf.float32)
y=tf.placeholder(shape=(None,10),dtype=tf.float32)

theta=tf.Variable(tf.ones(shape=(2,10)))
b=tf.Variable(tf.zeros(shape=(10)))

y_pre=tf.matmul(x,theta)+b
y_pre_softmax=tf.nn.softmax(y_pre)

loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pre_softmax)

optm=tf.train.GradientDescentOptimizer(learning_rate=0.01)
opt=optm.minimize(loss=loss)


correct_prediction = tf.equal(tf.cast(tf.argmax(y_pre_softmax, 1), tf.float32), tf.cast(tf.argmax(y, 1),tf.float32))

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


ini=tf.global_variables_initializer()

with tf.Session() as sess:
    ini.run()
    for n_epoche in range(n_epoches):
        opt.run(feed_dict={x:x_,y:y_})
        acc_eval=acc.eval(feed_dict={x:x_,y:y_})
        print(acc_eval)


