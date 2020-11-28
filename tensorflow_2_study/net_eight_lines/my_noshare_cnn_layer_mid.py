import tensorflow as tf
import  numpy as np

class My_parse_cnn_layer(tf.keras.layers.Layer):
    '''
    不共享的卷积，输出结果还是与输入维度一致，以便多层卷积
    '''
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',**kwargs):
        super(My_parse_cnn_layer, self).__init__(name='My_parse_cnn_layer',**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.units=kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding


    def build(self, input_shape):
        self.my_input_shape=input_shape
        self.w = []
        self.b = []
        # 这里的input只处理单通道，即(N,W,H,C)=(N,W,H,1)
        tmp_w=self.tmp_w = input_shape[1] // self.strides[0]
        tmp_h=self.tmp_h = input_shape[2] // self.strides[1]
        print('tmp_w:',tmp_w)
        print('tmp_h:',tmp_h)
        for i in range(tmp_w):
            tmp_weight=[]
            tmp_b=[]
            for j in range(tmp_h):
                tmp_weight.append(self.add_weight(
                    shape=(self.units,),
                    initializer="random_normal",
                    trainable=True,
                ))
                tmp_b.append(self.add_weight(
                    shape=(self.units,), initializer="random_normal", trainable=True
                ))
            self.w.append(tmp_weight)
            self.b.append(tmp_b)
    def call(self, inputs):
        split_y=[]

        for i in range(self.tmp_w):
            split_y_i=[]
            for j in range(self.tmp_h):
                tmp=self.my_draw(inputs,i,j)
                w=self.w[i][j]
                w=tf.reshape(w,(-1,1))
                mysum=tf.reduce_sum(tf.matmul(tmp,w)+self.b[i][j],axis=1)
                split_y_i.append( tf.reshape(mysum,(-1,1)))
            split_y.append(split_y_i)
        # 先按行拼接成矩阵(W,H,N,C)，再变换维度成(N,W,H,C)
        y=tf.convert_to_tensor(split_y)
        y=tf.transpose(y,(2,0,1,3))
        print(y.shape)
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