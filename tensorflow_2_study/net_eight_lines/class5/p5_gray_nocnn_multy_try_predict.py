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
from tensorflow_2_study.net_eight_lines.my_noshare_cnn_layer_mid import My_parse_cnn_layer as noshare_cnn
# 将RGB转灰度图

class jumpDenseBlock(Model):
    def __init__(self,units,jump_step,*args, **kwargs):
        self.jump_step=jump_step
        super().__init__(*args, **kwargs)
        self.d=tf.keras.layers.Dense(units=units,activation='relu')
        self.denseSeq=tf.keras.models.Sequential()
        if(jump_step>1):
            for i in range(jump_step-2):
                self.denseSeq.add(tf.keras.layers.Dense(units=units, activation='relu'))
        self.b=tf.keras.layers.BatchNormalization()
        self.a_last = Activation('relu')
    def call(self,inputs):
        x0=self.d(inputs)
        x1=x0
        if(self.jump_step>1):
            x1=self.denseSeq(x1)
        y=x1+x0
        y=self.b(y)
        return self.a_last(y)

class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.noshare_cnn1=noshare_cnn(1,(5,5),(1,1))
        self.noshare_cnn_flatten=My_parse_cnn_layer(1, (3, 3), (1, 1))
        # self.B=BatchNormalization()
        # self.A=Activation(activation='relu')
        self.B2 = BatchNormalization()
        self.A2 = Activation(activation='relu')
        self.jd1=jumpDenseBlock(units=128,jump_step=2)
        # self.dr1=tf.keras.layers.Dropout(0.2)
        self.jd2=jumpDenseBlock(units=256,jump_step=3)
        # self.dr2=tf.keras.layers.Dropout(0.2)
        self.jd3=jumpDenseBlock(units=512,jump_step=4)
        # self.jd3=jumpDenseBlock(units=512,jump_step=4)
        # self.jd3=jumpDenseBlock(units=512,jump_step=4)
        self.Dense=tf.keras.layers.Dense(units=10,activation='softmax')
    def call(self,inputs):
        x=inputs
        # x=self.noshare_cnn1(x)
        # x = self.B(x)
        # x = self.A(x)
        # 进入dense前，不能少了拉直
        x=self.noshare_cnn_flatten(x)
        x=self.B2(x)
        x=self.A2(x)
        x=self.jd1(x)
        # x=self.dr1(x)
        x=self.jd2(x)
        # x=self.dr2(x)
        x=self.jd3(x)
        y=self.Dense(x)
        return y
model = MyModel()

# 断点续训
check_point_path='./p5_gray_nocnn_multy_try/20201210/mnist.ckpt'
model.load_weights(check_point_path)
model.build((1,32,32,1))
# 打印网络
model.summary()



