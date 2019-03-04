'''
思想很重要，算法是实现思想，得出结果，实现必须执行
tf：
1.首先画出计算图
2.图的每一个节点构建，可能是constant，可能是varible，可能是placeholder
3.在图中使用解决思想的优化方法，进行迭代
4.真正在session中,给数据并进行计算，得出结果
5.评估结果

图画着画着就大概明白意思了，但还是需要全部进行学习
https://www.cnblogs.com/fydeblog/p/7429344.html
'''
import tensorflow as tf
import numpy as np

X_data=np.random.rand(100,1).astype(np.float32)
X_data.reshape(-1,1)
print(X_data.shape)

with tf.name_scope('X'):
    X=tf.placeholder(dtype=tf.float32,shape=(None,1),name='X')

with tf.name_scope('y'):
    y=X*0.3+5

with tf.name_scope('var'):
    w=tf.Variable(tf.random_normal((1,1)),name='w')
    tf.summary.histogram(name='w',values=w)
    b=tf.Variable(tf.zeros([1]),name='b')
    tf.summary.histogram(name='b',values=b)

with tf.name_scope('y_'):
    y_=tf.matmul(X,w)+b

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.square(y-y_))
    tf.summary.scalar(name='loss',tensor=loss)
with tf.name_scope('optimizer'):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)

with tf.name_scope('train'):
    train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
n_epochs=10000

with tf.Session() as sess:
    merge=tf.summary.merge_all()
    fw=tf.summary.FileWriter('logs/',sess.graph)
    init.run()
    for n in range(n_epochs):
        sess.run([train],feed_dict={X:X_data})
        # fw.run(feed_dict={X:X_data})
        if(n%100==0):
            _,summary_str=sess.run([train,merge], feed_dict={X: X_data})
            fw.add_summary(summary=summary_str,global_step=n)
            print('第',n,'次：')
            print(w.eval())
            print(b.eval())
            print()
    fw.close()