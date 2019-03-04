'''

使用keras一样的，先有思想，然后画出网络拓扑图，然后用api组建，最后进行训练、评估
又遇到loss乱跳、acc乱跳的情况了！！！！真的是好烦啊！！！！

'''
from MNIST import mnist_study
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Dense,Conv2D,ReLU,MaxPooling2D,Dropout,Flatten
import numpy as np
import matplotlib.pyplot as plt
# (X_train,y_train),(X_test,y_test)=mnist.load_data('D:\\学习笔记\\ai\\ai_study\\MNIST')
X_train,y_train=mnist_study.load_mnist(
    'D:\\学习笔记\\ai\\ai_study\\tensorflow_study\\CNN\\MNIST_DATA_BAK\\',kind='train')
X_test,y_test=mnist_study.load_mnist(
    'D:\\学习笔记\\ai\\ai_study\\tensorflow_study\\CNN\\MNIST_DATA_BAK\\',kind='t10k')
X_train=np.array(X_train)
X_train=X_train.reshape(-1,28,28,1)
X_test=np.array(X_test).reshape(-1,28,28,1)
print(X_train[2].shape)
print(y_train[2])

'''
def label_onehot(label):
    label_=np.zeros(shape=10)
    label_[label]=1
    return label_
y_train=np.array([label_onehot(y_train[i]) for i in range(y_train.shape[0])])
print(y_train.shape)
y_test=np.array([label_onehot(y_test[i]) for i in range(y_test.shape[0])])
input=Input(shape=(28,28,1))
h=Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME')(input)
h=MaxPooling2D(pool_size=(2,2))(h)
h=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='SAME')(h)
h=MaxPooling2D(pool_size=(2,2))(h)
h=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='SAME')(h)
h=MaxPooling2D(pool_size=(2,2))(h)
h=Flatten()(h)
h=Dense(128,activation='relu')(h)
h=Dense(64,activation='relu')(h)
h=Dense(32,activation='relu')(h)
output=Dense(10,activation='softmax')(h)

model=Model(inputs=input,outputs=output)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=X_train,y=y_train,batch_size=128,epochs=5,validation_data=(X_test,y_test))
'''