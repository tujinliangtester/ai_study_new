'''
TensorBoard可以画出tf中的节点及流转关系，即可以画出tf的图
用summary中的FileWriter将tf的图保存下来即可


'''


import tensorflow as tf

with tf.name_scope('graphabc'):
    A=tf.constant([[3,3]],name='A')
    B=tf.constant([[3],[3]],name='B')
    C=tf.matmul(A,B,name='C')

with tf.Session() as sess:
    writer=tf.summary.FileWriter('logs/',sess.graph)
    inti=tf.global_variables_initializer()
    inti.run()