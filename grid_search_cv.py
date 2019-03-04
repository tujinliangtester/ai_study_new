'''
其实这里面包含两方面的东西：
1.模型验证，通过对dataset的选取，依次进行训练和测试验证；
    同时，验证的方式为误差平方再求和，再平均
    这里面，会选择所有当中，误差最小的那个模型
    实际上，如果是K折交叉验证，则会有K个模型
2.超参数的组合，对每一个超参数组合，采用1步骤进行求解模型，并得出该组合的平均误差
    选择平均误差最小的组合作为模型的超参数
    实际上，得出的最优超参数组合，不一定是最优模型的情况，因为这里是将K个模型进行了加和，
    可能取不到真正的误差最小的超参数组合。
    当然了，这其实也是一种防止过拟合的情况，因为单个测试集也算是模型训练过程的一部分，应该防止过拟合。
'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

iris=load_iris()
X=iris['data']
Y=iris['target']

grid_search={'max_iter':[100,1000,1000],
             'C':[1.0,0.9,0.8],
             'tol':[1e-3,1e-4,1e-2]
             }
clf=LogisticRegression(solver='sag')
grid_search_clf=GridSearchCV(clf,param_grid=grid_search,cv=5)
grid_search_clf.fit(X,Y)
print(grid_search_clf.best_estimator_)
x=X[100:101,:]
print(grid_search_clf.predict(x))