# import
import tensorflow as tf
import numpy as np
import os

from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import BatchNormalization, Activation, Dense, Dropout
from tensorflow.python.keras.layers.convolutional import Conv, Conv2D


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

class resBlock(Model):
    def __init__(self, filters,kernel_size,strides,*args, **kwargs):
        self.strides=strides
        self.bo=False
        super().__init__(*args, **kwargs)
        self.c1=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')
        self.b1=BatchNormalization()
        self.a1=Activation('relu')

        self.c2=Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')
        self.b2=BatchNormalization()
        self.a2=Activation('relu')
        if(strides>1):
            self.bo=True
            #    注意，这里layer只能在init函数中创建，否则tf会无法创建变量
            self.c_down=Conv2D(filters=1,kernel_size=(1,1),strides=self.strides,padding='same')
        #由于训练模型后期测试准确率不再提高，增加dropout尝试
        # self.d=Dropout(0.2)
    def call(self,inputs):
        x0=inputs
        x1=self.c1(x0)
        x1=self.b1(x1)
        x1=self.a1(x1)

        x2=self.c2(x1)
        x2=self.b2(x2)
        if(self.bo):
            x0=self.c_down(x0)
        y=x2+x0
        y=self.a2(y)
        # y=self.d(y)
        return y

class resNet18(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c1=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same')
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.resNetBlock=tf.keras.models.Sequential()
        for i in range(2):
            resB=resBlock(filters=64,kernel_size=(3,3),strides=1)
            self.resNetBlock.add(resB)
        resB = resBlock(filters=128, kernel_size=(3, 3), strides=2)
        self.resNetBlock.add(resB)

        resB = resBlock(filters=128, kernel_size=(3, 3), strides=1)
        self.resNetBlock.add(resB)

        resB = resBlock(filters=256, kernel_size=(3, 3), strides=2)
        self.resNetBlock.add(resB)

        resB = resBlock(filters=256, kernel_size=(3, 3), strides=1)
        self.resNetBlock.add(resB)

        resB = resBlock(filters=512, kernel_size=(3, 3), strides=2)
        self.resNetBlock.add(resB)

        resB = resBlock(filters=512, kernel_size=(3, 3), strides=1)
        self.resNetBlock.add(resB)

        self.p=tf.keras.layers.GlobalAveragePooling2D()
        self.dens1=Dense(units=100,activation='relu')
        self.d1=tf.keras.layers.Dropout(0.2)
        self.dens2=Dense(units=100,activation='relu')
        self.d2=tf.keras.layers.Dropout(0.2)
        self.dens=Dense(units=10,activation='softmax')
    def call(self,x):
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.resNetBlock(x)
        x=self.p(x)
        x=self.dens1(x)
        x=self.d1(x)
        x=self.dens2(x)
        x=self.d2(x)
        x=self.dens(x)
        return x

model = resNet18()

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
    x_train, y_train, batch_size=256, epochs=20,
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
# 保存参数，方便画出整体图形
with open('history.txt','a') as f:
    f.write('acc:')
    f.write(history.history['sparse_categorical_accuracy'])
    f.write('val_acc:')
    f.write(history.history['val_sparse_categorical_accuracy'])
    f.write('loss:')
    f.write(history.history['loss'])
    f.write('val_loss:')
    f.write(history.history['val_loss'])

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