'''
1.数据预处理
    数据离散化(x和y都需要进行，一般情况下模型会自动进行处理)
2.建模思想
    决策树、随机森林
    决策树的思想其实就是根据非常多的条件（与离散化处理有关）逐级进行判断并决策
    随机森林的处理，包括记录数量和维度选取
3.求解方式
    纯度：基尼系数、熵、方差
    用纯度来选择分裂维度（可以反过来反映各维度对结果的相关性）

    树层数、叶子数、叶子记录数、分裂点记录数等
    针对上述的各种指标，可以进行前剪枝、后剪枝，防止过拟合
4.模型评估
    准确率
'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomTreesEmbedding
from sklearn import tree
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()
X=iris['data']#[:,:2]#选取相关性较低的特征，以试验模型的错误率效果更好
y=iris['target']
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)

max_depth=4
tree_clf_gini=tree.DecisionTreeClassifier(random_state=23,criterion="gini",max_depth=max_depth)
tree_clf_entropy=tree.DecisionTreeClassifier(random_state=23,criterion="entropy",max_depth=max_depth)
tree_clf_gini.fit(X_train,y_train)
tree_clf_entropy.fit(X_train,y_train)
y_predict_gini=tree_clf_gini.predict(X_test)
y_predict_entropy=tree_clf_entropy.predict(X_test)
acc_gini=accuracy_score(y_predict_gini,y_test)
acc_ent=accuracy_score(y_predict_entropy,y_test)
print(acc_gini,acc_ent)

depth=np.arange(1,20)
acc_list=[]
for i in depth:
    tree_clf_gini = tree.DecisionTreeClassifier(random_state=23, criterion="gini", max_depth=i)
    tree_clf_gini.fit(X_train, y_train)
    y_predict_gini = tree_clf_gini.predict(X_test)
    acc_gini = accuracy_score(y_predict_gini, y_test)
    acc_list.append(acc_gini)

plt.plot(depth,acc_list,'ro-')
plt.show()