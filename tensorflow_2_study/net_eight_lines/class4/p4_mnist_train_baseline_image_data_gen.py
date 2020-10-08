# import
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def my_load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# 准备数据集
mnist = tf.keras.datasets.mnist
path = 'C:/Users/Administrator\Downloads\Deeplearning-master/mnist.npz'
(x_train, y_train), (x_test, y_test) = my_load_data(path)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 数据增强
image_train = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.5,
    rescale=1
)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
image_train.fit(x_train)

# 搭建网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(10, 'softmax'),
])

# 定义网络
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 训练网络
model.fit(
    image_train.flow(x_train, y_train, batch_size=32), epochs=5,
    validation_data=(x_test, y_test), validation_steps=1
)

# 打印网络
model.summary()
