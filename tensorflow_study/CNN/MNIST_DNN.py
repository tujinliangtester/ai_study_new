from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# train (55000, 784)
# test (10000, 784)
# validation (5000, 784)
mnist=input_data.read_data_sets('MNIST_DATA_BAK/',one_hot=True)
print(mnist.validation.labels.shape)
m,n=mnist.train.images.shape

# 构建图
x=tf.placeholder(dtype=tf.float32,shape=(None,n))
y=tf.placeholder(dtype=tf.float32,shape=(None,10))

theta=tf.Variable(initial_value=tf.ones(shape=(n,10),dtype=tf.float32))
b=tf.Variable(initial_value=tf.zeros(shape=(1,10),dtype=tf.float32))

y_hat=tf.matmul(x,theta)+b
y_hat=tf.nn.softmax(y_hat)

loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat)

optm=tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt=optm.minimize(loss)

# acc=tf.equal(tf.arg_max(y,1),tf.arg_max(y_hat,1))

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1))
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(10000):
        x_train_batch,y_train_batch=mnist.train.next_batch(batch_size=100)
        opt.run(feed_dict={x:x_train_batch,y:y_train_batch})
        if(i%1000==0):
            print('train i:',i,'acc:',acc.eval(feed_dict={x:x_train_batch,y:y_train_batch}))
            print('valid i:',i,'acc:',acc.eval(feed_dict={x:mnist.validation.images,y:mnist.validation.labels}))
    print('test acc:',acc.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))
