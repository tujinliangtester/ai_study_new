'''
在图形识别时，将图片某一小块作为一个部分，来提取特征，要比直接转换位一维数组要好
自然而然的，就可以想到这样来做（仍然离不开算法来源于思想）
卷积核是怎么想出来的？卷积核是对图片不同角度的审视（其实这在1998年就已经有成熟的理论和实践了。。。）
其实，卷积核也可以看做神经网络中的一部分

池化：有两种处理方式，一是取最大，相当于锐化处理，二是取平均相当于模糊处理
一般识别都是取最大，当然还得看实际模型的预测结果，科学是实践出来的

'''
from sklearn.datasets import load_sample_images
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

dataset=np.array(load_sample_images()['images'])
print(dataset.shape)
batch_size,h,w,channels=dataset.shape
filters=np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters[3,:,:,0]=1
filters[:,3,:,1]=1
'''
strides:
default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
'''
X=tf.placeholder(shape=dataset.shape,name='X',dtype=np.float32)
# convolution=tf.nn.conv2d(input=X,filter=filters,strides=[1,2,2,1],padding='SAME')
convolution=tf.nn.conv2d(input=X,filter=filters,strides=[1,2,2,1],padding='VALID')

with tf.Session() as sess:
    output=sess.run(convolution,feed_dict={X:dataset})

n=0
plt.imshow(dataset[n])
plt.show()
plt.imshow(output[n,:,:,1])
plt.show()
plt.imshow(output[n,:,:,0])
plt.show()
