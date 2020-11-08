import tensorflow as tf


class Mul_dimen_layer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Mul_dimen_layer, self).__init__()
        self.units = units
        self.a=tf.keras.layers.Activation(activation='relu')

    def build(self, input_shape):
        self.w = []
        self.b = []
        for i in range(12):
            self.w.append(self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="random_normal",
                trainable=True,
            ))
            self.b.append(self.add_weight(
                shape=(self.units,), initializer="random_normal", trainable=True
            ))

    def call(self, inputs):
        x0=tf.matmul(inputs,self.w[0])+self.b[0]
        x1=tf.matmul(inputs,self.w[1])+self.b[1]
        x2=tf.matmul(inputs,self.w[2])+self.b[2]
        #为了简单起见，所有的矩阵都是用正方形!
        x3=tf.matmul(x0,self.w[3])+self.b[3]
        x4=tf.matmul(x0,self.w[4])+self.b[4]
        x5=tf.matmul(x1,self.w[5])+self.b[5]
        x6=tf.matmul(x1,self.w[6])+self.b[6]
        x7=tf.matmul(x2,self.w[7])+self.b[7]
        x8=tf.matmul(x2,self.w[8])+self.b[8]
        x9=x3+x5
        x9 = tf.matmul(x9, self.w[9]) + self.b[9]
        x10=x4+x7
        x10=tf.matmul(x10,self.w[10])+self.b[10]
        x11=x6+x8
        x11=tf.matmul(x11,self.w[11])+self.b[11]
        y=x9+x10+x11
        return self.a(y)

