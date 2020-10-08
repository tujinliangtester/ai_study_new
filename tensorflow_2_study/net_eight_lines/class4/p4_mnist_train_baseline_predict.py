# import
import tensorflow as tf
from PIL import Image
import numpy as np

# 复现网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(10, 'softmax'),
])

# 加载模型参数
check_point_path='./check_point/mnist.ckpt'
model.load_weights(check_point_path)

while(True):
    bool=input('是否继续？y/n')
    if(bool!='n'):
        # 处理待预测数据，保持与训练、测试数据各方面一致
        # 需要明确的知道原数据都有哪些特征，其实在数据清洗过程中就比较清楚了
        img='predict_img/'+input('输入图片地址')
        img=Image.open(img)
        img.resize((28,28),Image.ANTIALIAS)
        img_arr=np.array(img.convert('L'))
        img_arr=img_arr/255.0
        img_arr=img_arr[np.newaxis,...]

        # 预测
        result=model.predict(img_arr)
        pred=tf.argmax(result,axis=1)
        print('预测的值为：',pred)
    else:
        break