'''
1.数据预处理
    数据离散化(x和y都需要进行，一般情况下模型会自动进行处理)
2.建模思想
    决策树、随机森林
    决策树的思想其实就是根据非常多的条件（与离散化处理有关）逐级进行判断并决策
    随机森林的处理，包括记录数量和维度选取,是一种bagging的思想
    bagging思想不仅仅是用于随机森林，也包含多种方式进行预测，然后少数服从多数
3.求解方式
    纯度：基尼系数、熵、方差
    用纯度来选择分裂维度（可以反过来反映各维度对结果的相关性）

    树层数、叶子数、叶子记录数、分裂点记录数等
    针对上述的各种指标，可以进行前剪枝、后剪枝，防止过拟合
4.模型评估
    准确率

剩下的部分：
1.随机森林回归（可以不做）
2.bagging的拓展，sklearn官网，选取时除了课程中的三种方式外，还需要加一个神经网络

'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import bagging
from sklearn import tree
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

iris=load_iris()
X=iris['data']#[:,:2]#选取相关性较低的特征，以试验模型的错误率效果更好
y=iris['target']
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)

n_estimators=1
rfc=RandomForestClassifier(n_estimators=n_estimators, random_state=23,max_depth=4)
rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
acc=accuracy_score(y_test,y_predict)
print(acc)

n_estimators=np.arange(1,20)
acc_list=[]
for i in n_estimators:
    rfc = RandomForestClassifier(n_estimators=i, random_state=23, max_depth=4)
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    acc_list.append(acc)
plt.plot(n_estimators,acc_list,'ro-')
plt.show()

acc_list=[]
rfc=RandomForestClassifier(random_state=23)
param_grid={'n_estimators':[1,2,3,4,5,6,7,8,9],
            'max_depth':[1,2,3,4,5,6,7,8,9]}
grid_rfc=GridSearchCV(rfc,param_grid=param_grid,cv=5)
grid_rfc.fit(X_train,y_train)
y_predict = grid_rfc.predict(X_test)
acc = accuracy_score(y_test, y_predict)
acc_list.append(acc)
print(acc)