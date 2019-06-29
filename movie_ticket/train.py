

import tensorflow as tf
from  movie_ticket import  tmp


n_epoches=100

x_=tmp.read_data('x.npy')
y_=tmp.read_data('y.npy')

x=tf.placeholder(shape=(None,2),dtype=tf.float32,name='x')
y=tf.placeholder(shape=(None,1),dtype=tf.float32,name='y')

theta=tf.Variable(tf.ones(shape=(2,1)),name='theta')
b=tf.Variable(tf.zeros(shape=(1)),name='b')

tf.summary.histogram(name='theta',values=theta)
tf.summary.histogram(name='b',values=b)

y_pre=tf.matmul(x,theta)+b
# y_pre_softmax=tf.nn.softmax(y_pre)

# loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pre_softmax,name='loss')

loss = tf.reduce_mean(tf.square(y_ - y_pre))
tf.summary.scalar(name='loss',tensor=loss)

optm=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
opt=optm.minimize(loss=loss)


# correct_prediction = tf.equal(tf.cast(tf.argmax(y_pre_softmax, 1), tf.float32), tf.cast(tf.argmax(y, 1),tf.float32))
#
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='acc')


ini=tf.global_variables_initializer()

with tf.Session() as sess:
    merge = tf.summary.merge_all()
    ini.run()
    fw=tf.summary.FileWriter('logs/',sess.graph)
    for n_epoche in range(n_epoches):
        _,summary_str=sess.run([opt,merge],feed_dict={x:x_,y:y_})
        fw.add_summary(summary=summary_str, global_step=n_epoche)

        y_pre_eval=y_pre.eval(feed_dict={x:x_,y:y_})
        # print(len(y_pre_eval))
        print('y_pre_eval',y_pre_eval)
    fw.close()


