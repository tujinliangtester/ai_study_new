# import
import tensorflow as tf
import numpy as np
import os

from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense


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


# 数据增强
image_train = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.5,
    rescale=1
)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# image_train.fit(x_train)

# 搭建网络

class ConvBnRelu(Model):
    def __init__(self,ch,kernal_size,strides=1,padding='same'):
        super(ConvBnRelu, self).__init__()
        self.model=tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(ch,kernal_size,strides,padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
        ])
    def call(self,x):
        return self.model(x)

class InceptionNet(Model):
    def __init__(self, strides, *args, **kwargs):
        super(InceptionNet,self).__init__(*args, **kwargs)
        self.c1=ConvBnRelu(16,(1,1),strides)
        self.c2_1=ConvBnRelu(16,(1,1),strides)
        self.c2_2=ConvBnRelu(16,(3,3),strides)
        self.c3_1=ConvBnRelu(16,(1,1),strides)
        self.c3_2=ConvBnRelu(16,(5,5),strides)
        self.c4_1=tf.keras.models.Sequential([
            tf.keras.layers.MaxPool2D((3, 3), 1,'same')])
        self.c4_2=ConvBnRelu(16,(1,1),strides)
    def call(self,x):
        x1=self.c1(x)
        x2_1=self.c2_1(x)
        x2_2=self.c2_2(x2_1)
        x3_1=self.c3_1(x)
        x3_2=self.c3_2(x3_1)
        x4_1=self.c4_1(x)
        x4_2=self.c4_2(x4_1)
        # 在深度方向叠加，NWHC，即3为深度方向
        x=tf.concat([x1,x2_2,x3_2,x4_2],axis=3)
        return x

class Inception10(Model):
    def __init__(self, initCnns,inceptionBlocks,denses):
        super(Inception10,self).__init__()
        self.c1=ConvBnRelu(16,(3,3))
        self.inceptionBlock=tf.keras.models.Sequential()
        for i in range(inceptionBlocks):
            for j in range(2):
                if(j==0):
                    block=InceptionNet(strides=2)
                else:
                    block=InceptionNet(strides=1)
                self.inceptionBlock.add(block)
        self.d=Dense(denses,'softmax')
    def call(self,x):
        x1=self.c1(x)
        x2=self.inceptionBlock(x1)
        x_f=tf.keras.layers.Flatten(x2)
        x3=self.d(x_f)
        return x3

model=Inception10(16,2,10)

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
    x_train, y_train, batch_size=128, epochs=5,
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