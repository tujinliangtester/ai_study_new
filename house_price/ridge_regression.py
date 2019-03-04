'''
问题思考，为什么预测结果差这么远？
用神经网络预测的结果也是，很离谱

可能的原因：
1.区域转换为分类号的时候不合适，分类号的大小未与区域的重要程度形成一定的关联，用线性模型的话可能就会很有问题
    包括用多项式的线性回归也会很有问题
    其实可以考虑随机森林？分类问题，或者因素的分类对结果有明显影响的场景，用随机森林可能效果会不错
2.还有可能的原因是，影响房价的关键因素，除了区域外，还有具体的位置、楼盘的新旧、装修程度、周围设施等
    数据是第一步，如果第一步失败，则后续几乎不可能成功，而在人工智能这一步来看，必然会失败
    往往在准备数据的时候，是否应该找尽量多的维度？
'''

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import *


scaler = StandardScaler()
data = pd.read_csv('house_price.csv', encoding="utf-8")
data = np.array(data)
y = data[:, 3:4]

# 将分类转换为分类号
enc = OrdinalEncoder()
enc.fit(data[:, 2:3])
data_enc = enc.transform(data[:, 2:3])
# print('data_enc:',data_enc)
data = hstack((data[:, :2], data_enc))

X = data[:, :3]
#归一化
scaler.fit(X)
X = scaler.transform(X)

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=23)
X_train, X_test, y_train, y_test = X[:300, :], X[300:, :], y[:300, :], y[300:, :]
print('X_train:', hstack((X_train, y_train)))
# print('y_train:',y_train)


# model=Ridge(solver='sag',random_state=23)
# model.fit(X,y)
# y_predict=model.predict(X_test)
# print(y_predict)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
# print(X_poly)
model2 = Ridge(solver='sag', random_state=23, tol=1e-4, max_iter=10000)
model2.fit(X_poly, y_train)
# X_test=poly_features.fit_transform(X_test)
y_predict = model2.predict(X_poly)

print(hstack((y_train, y_predict)))
print(model2.coef_)
print(model2.intercept_)


