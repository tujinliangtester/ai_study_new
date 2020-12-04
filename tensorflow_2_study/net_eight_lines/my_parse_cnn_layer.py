import tensorflow as tf
import  numpy as np

class My_parse_cnn_layer(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',**kwargs):
        super(My_parse_cnn_layer, self).__init__(name='MyParseCnn',**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.units=kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding


    def build(self, input_shape):
        self.my_input_shape=input_shape
        # 这里的input只处理单通道，即(N,W,H,C)=(N,W,H,1)
        self.tmp_w = input_shape[1] // self.strides[0]
        self.tmp_h = input_shape[2] // self.strides[1]
        self.total_num=self.tmp_h*self.tmp_w
        self.total_num_units=self.total_num*self.units
        #我在这里加了个名字，居然就可以了？？？！！！
        self.w=self.add_weight(
                    name='w',
                    shape=(self.total_num_units,),
                    initializer="random_normal",
                    trainable=True,
                )
        self.b= self.add_weight(
            name='b',
            shape=(self.total_num_units,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        split_y=[]
        for i in range(self.tmp_w):
            for j in range(self.tmp_h):
                tmp=self.my_draw(inputs,i,j)
                #取w和b
                start=i*self.tmp_h*self.units+j*self.units
                end=start+self.units
                mysum=tf.reduce_sum(
                    tf.matmul(tmp,tf.reshape(self.w[start:end],(-1,1)))
                        +self.b[start:end],
                    axis=1)
                split_y.append( tf.reshape(mysum,(-1,1)))
        y=tf.concat(split_y,axis=-1)
        return y

    def my_draw(self,inputs,i,j):
        tmp_i=i-1
        tmp_j=j-1
        res= None
        for m in range(self.kernel_size[0]):
            tmp_i+=1
            for n in range(self.kernel_size[1]):
                tmp_j += 1
                if(self.my_juge_out(inputs,tmp_i,tmp_j)):
                    # todo 报错！先写死一个批次的数量
                    # 这里暂时的思路是通过内置函数来增加维度，并且用0填充
                    tmp=np.zeros(shape=(100,1))
                else:
                    tmp=inputs[:,tmp_j,tmp_j]
                if(res is None):
                    res=tmp
                else:
                    res=tf.concat([res,tmp],axis=-1)
        return res
    def my_juge_out(self,inputs,i,j):
        if(i<0 or j<0 or i>=inputs.shape[1] or j>=inputs.shape[2]):
            return True
        return False

if __name__=='__main__':
    w=tf.ones_initializer()
    w=w((1,9))
    print(w)
    w=tf.transpose(w)
    print(w)