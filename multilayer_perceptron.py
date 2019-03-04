import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
# //作业是MLPRegressor预测水泥强度
# ps 画出神经网络拓扑图
# 下面是MLPRegressor与MLPClassifier的区别
# Class MLPRegressor implements a multi-layer perceptron (MLP)
# that trains using backpropagation with no activation function in the output layer,
#  which can also be seen as using the identity function as activation function.
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
'''
sklearn.com
 The key step is computing the partial derivatives above.
 backpropagation gives an efficient way to compute these partial derivatives.
 这个求偏导的算法并没看懂。。。
 有时间的话可以再研究研究


'''
'''
Multi-layer Perceptron is sensitive to feature scaling, 
so it is highly recommended to scale your data. 
For example, scale each attribute on the input vector X to [0, 1] or [-1, +1], 
or standardize it to have mean 0 and variance 1. 
Note that you must apply the same scaling to the test set for meaningful results. 
You can use StandardScaler for standardization.

>>>
>>> from sklearn.preprocessing import StandardScaler  
>>> scaler = StandardScaler()  
>>> # Don't cheat - fit only on training data
>>> scaler.fit(X_train)  
>>> X_train = scaler.transform(X_train)  
>>> # apply same transformation to test data
>>> X_test = scaler.transform(X_test)  

'''

'''
X=[[1,1],[0,0],[3,4],[5,0]]
Y=[1,0,0,1]
clf=MLPClassifier(hidden_layer_sizes=(2,2),activation='logistic',
                  alpha=0.0001,max_iter=10000,solver='sgd')
clf.fit(X,Y)

X_test=[[1,0],[5,1]]
Y_predict=clf.predict(X_test)
Y_proba=clf.predict_proba(X_test)
print(Y_predict)
print(Y_proba)
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
'''

# 开始预测波士顿房价
print('HousePrice')
data = pd.read_csv('HousePrice.csv')

data = np.array(data)
data_test = data[271:, :]
data = data[:270, :]  # 270 for train
X = data[:, :13]
scaler.fit(X)
X = scaler.transform(X)
Y = data[:, 13]  # 注意，这里的13后面一定不能加冒号！！！否则Y会变成2D矩阵，交给模型算的时候就完全不一样了！！！
model = MLPRegressor(hidden_layer_sizes=(200, 200,200,200,200), activation='relu', solver='sgd', max_iter=1000)
model.fit(X, Y)
print('coef.shape',[coef.shape for coef in model.coefs_])
print([coef for coef in model.coefs_])

y_hat = model.predict(X)
X_test = data_test[:, :13]
X_test = scaler.transform(X_test)
x = X_test[:, 6:7]
Y_test = data_test[:, 13]

Y_predict = model.predict(X_test)

print(Y_test)
print('Y_predict', Y_predict)
# print(np.corrcoef(Y_test,Y_predict))
# print(Y.shape)
'''
做的時候發現有時候出來的結果是完全一樣的，即因素變了，結果卻沒有變化；但有時候結果又是正確的
1.sklearn也說了，這個模型MLP的損失函數不是凸函數，即有多個本地最小值，需要多次嘗試。
但，如果有多個本地最小值，就是導致計算結果完全一樣的原因嗎？表示有點懷疑。
2.同時，該模型對因素取值範圍敏感，最好是進行歸一化等預處理，這可能也是導致出現該異常的一個原因
'''
plt.plot(x, Y_test, 'b.')
plt.plot(x, Y_predict, 'r.')
plt.show()

'''
# 由于出现了预测结果完全一样的异常，所以先用简单的数据进行验证
print('MLPRegressor for easier')
X=[[1,1],[0,0],[3,4],[5,0]]
Y=[1,3,0,1]
clf=MLPRegressor(hidden_layer_sizes=(2,2),activation='logistic',
                  alpha=0.0001,max_iter=10000,solver='sgd')
clf.fit(X,Y)

X_test=[[1,0],[5,1]]
Y_predict=clf.predict(X_test)
print(Y_predict)
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
'''
