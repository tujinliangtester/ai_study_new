# import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def my_load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# 准备数据集
mnist = tf.keras.datasets.mnist
path='D:\BaiduNetdiskDownload/mnist.npz'
(x_train, y_train), (x_test, y_test) = my_load_data(path)
x_train = x_train / 255.0
x_test = x_test / 255.0


class Mul_dimen_layer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Mul_dimen_layer, self).__init__()
        self.units = units
        self.a=tf.keras.layers.Activation(activation='relu')

    def build(self, input_shape):
        self.w = []
        self.b = []
        for i in range(12):
            self.w.append(self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            ))
            self.b.append(self.add_weight(
                shape=(self.units,), initializer="random_normal", trainable=True
            ))

    def call(self, inputs):
        x0=tf.matmul(inputs,self.w[0])+self.b[0]
        x1=tf.matmul(inputs,self.w[1])+self.b[1]
        x2=tf.matmul(inputs,self.w[2])+self.b[2]
        #为了简单起见，所有的矩阵都是用正方形!
        x3=tf.matmul(x0,self.w[3])+self.b[3]
        x4=tf.matmul(x0,self.w[4])+self.b[4]
        x5=tf.matmul(x1,self.w[5])+self.b[5]
        x6=tf.matmul(x1,self.w[6])+self.b[6]
        x7=tf.matmul(x2,self.w[7])+self.b[7]
        x8=tf.matmul(x2,self.w[8])+self.b[8]
        x9=x3+x5
        x9 = tf.matmul(x9, self.w[9]) + self.b[9]
        x10=x4+x7
        x10=tf.matmul(x10,self.w[10])+self.b[10]
        x11=x6+x8
        x11=tf.matmul(x11,self.w[11])+self.b[11]
        y=x9+x10+x11
        return self.a(y)


# 搭建网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(32, 'relu'),
    Mul_dimen_layer(),
    tf.keras.layers.Dense(10, 'softmax'),
])

# 定义网络
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 训练网络
history=model.fit(
    x_train, y_train, batch_size=32, epochs=5,
    validation_data=(x_test, y_test), validation_steps=1
)

# 打印网络
model.summary()


# 画图
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='acc')
plt.plot(val_acc,label='val acc')
plt.title('acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
plt.title('loss')
plt.legend()
plt.show()