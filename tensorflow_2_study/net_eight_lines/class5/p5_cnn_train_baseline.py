# import
import tensorflow as tf
import numpy as np
import os

from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

def my_load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 准备数据集
path='D:/学习笔记/ai_data_set/cifar-10-python/cifar-10-batches-py/data_batch_'
i=1
x_train, y_train=[],[]
while(i<6):
    newpath=path+str(i)
    f=unpickle(newpath)
    if(i==1):
        x_train = f[b'data']
        y_train =f[b'labels']
    else:
        x_train=tf.concat([x_train,f[b'data']],axis=0)
        y_train=tf.concat([y_train,f[b'labels']],axis=0)
    i+=1

x_train=tf.cast(x_train,tf.float32)
x_train=np.array(x_train).reshape((-1,3,32,32))
y_train=np.array(y_train).reshape(-1)
# one hot编码
# y_train=to_categorical(np.array(y_train))

path='D:/学习笔记/ai_data_set/cifar-10-python/cifar-10-batches-py/test_batch'
f=unpickle(path)
x_test, y_test=f[b'data'],f[b'labels']

x_test=tf.cast(x_test,tf.float32)
x_test=np.array(x_test).reshape((x_test.shape[0],3,32,32))
y_test=np.array(y_test)
# y_test=to_categorical(np.array(y_test))

# 官方数据集
# tf.keras.datasets.cifar10.load_data()


# 数据增强
image_train = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.5,
    rescale=1
)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# image_train.fit(x_train)

# 搭建网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D((2,2),2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(10, 'softmax'),
])
var = tf.keras.optimizers.Adam
# 定义网络
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 断点续训
check_point_path='./check_point/mnist.ckpt'
# if os.path.exists(check_point_path+'.index'):
#     print('加载已有模型参数，继续训练')
#     model.load_weights(check_point_path)

call_back=tf.keras.callbacks.ModelCheckpoint(
    filepath=check_point_path,
    save_weights_only=True,
    save_best_only=True
)

# 训练网络
history=model.fit(
    # image_train.flow(x_train, y_train, batch_size=32), epochs=5,
    x_train, y_train, batch_size=32, epochs=50,
    validation_data=(x_test, y_test), validation_steps=1,
    callbacks=call_back
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