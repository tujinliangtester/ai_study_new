# import
import tensorflow as tf
import numpy as np
import os

from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import BatchNormalization, Activation, Dense
from tensorflow.python.keras.layers.convolutional import Conv, Conv2D
from tensorflow_2_study.net_eight_lines.my_parse_cnn_layer import My_parse_cnn_layer
# 将RGB转灰度图

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
path='D:\BaiduNetdiskDownload\cifar-10-python\cifar-10-batches-py/data_batch_'
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
# 转换数据，将channel层放到最后，即NCWH 转成NWHC
x_train=tf.transpose(x_train,(0,2,3,1))
y_train=np.array(y_train).reshape(-1)
# one hot编码
# y_train=to_categorical(np.array(y_train))

path='D:\BaiduNetdiskDownload\cifar-10-python\cifar-10-batches-py/test_batch'
f=unpickle(path)
x_test, y_test=f[b'data'],f[b'labels']

x_test=tf.cast(x_test,tf.float32)
x_test=np.array(x_test).reshape((-1,3,32,32))
# 转换数据，将channel层放到最后，即NCWH 转成NWHC
x_test=tf.transpose(x_test,(0,2,3,1))
y_test=np.array(y_test)
# y_test=to_categorical(np.array(y_test))

# 官方数据集
# tf.keras.datasets.cifar10.load_data()

# 转灰度
x_train=tf.image.rgb_to_grayscale(x_train)
x_test=tf.image.rgb_to_grayscale(x_test)

# 数据增强
image_train = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.5,
    rescale=1
)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# image_train.fit(x_train)

class jumpDenseBlock(Model):
    def __init__(self,units,jump_step,*args, **kwargs):
        self.jump_step=jump_step
        super().__init__(*args, **kwargs)
        self.d=tf.keras.layers.Dense(units=units,activation='relu')
        self.denseSeq=tf.keras.models.Sequential()
        if(jump_step>1):
            for i in range(jump_step-2):
                self.denseSeq.add(tf.keras.layers.Dense(units=units, activation='relu'))
        self.a_last = Activation('relu')
    def call(self,inputs):
        x0=self.d(inputs)
        x1=x0
        if(self.jump_step>1):
            x1=self.denseSeq(x1)
        y=x1+x0
        return self.a_last(y)

class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_cnn=My_parse_cnn_layer(1,(3,3),(1,1))
        self.A=Activation(activation='relu')
        self.B=BatchNormalization()
        self.jd1=jumpDenseBlock(units=128,jump_step=2)
        self.jd2=jumpDenseBlock(units=256,jump_step=3)
        self.jd3=jumpDenseBlock(units=512,jump_step=4)
        self.jd3=jumpDenseBlock(units=512,jump_step=4)
        self.jd3=jumpDenseBlock(units=512,jump_step=4)
        self.Dense=tf.keras.layers.Dense(units=10,activation='softmax')
    def call(self,inputs):
        x=self.split_cnn(inputs)
        x=self.A(x)
        x=self.B(x)
        x=self.jd1(x)
        x=self.jd2(x)
        x=self.jd3(x)
        y=self.Dense(x)
        return y
model = MyModel()

# 定义网络
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 断点续训
check_point_path='./check_point_gray_nocnn/mnist.ckpt'
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
    # x_train[:300,:,:,:], y_train[:300], batch_size=100, epochs=1, #初步运行，试错
    x_train, y_train, batch_size=100, epochs=300,
    # validation_data=(x_test[:100], y_test[:100]), validation_steps=1,
    # callbacks=call_back
)

# 打印网络
model.summary()

# 画图
acc=history.history['sparse_categorical_accuracy']
# val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
# val_loss=history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='acc')
# plt.plot(val_acc,label='val acc')
plt.title('acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='loss')
# plt.plot(val_loss,label='val_loss')
plt.title('loss')
plt.legend()
plt.show()


# 结论：经过试验，发现效果并没有提升
