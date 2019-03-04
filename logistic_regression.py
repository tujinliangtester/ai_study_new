import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()
print(list(iris.keys()))
# print(iris['data'])
print(iris['feature_names'])
'''
['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
'''
X=iris['data']
print(X)
Y=iris['target']
print(Y)
LGmodel=LogisticRegression(multi_class='multinomial',solver='sag')
LGmodel.fit(X,Y)

X_new=np.linspace(0,3,1000).reshape(-4,4)
Y_proba=LGmodel.predict_proba(X=X_new)
print(Y_proba)

plt.plot(X_new,Y_proba[:,0],'r.')
plt.plot(X_new,Y_proba[:,1],'g.')
plt.plot(X_new,Y_proba[:,2],'b.')
# plt.axis
# plt.show()

X_new=np.array([[1,1,1,1]])
Y_proba=LGmodel.predict_proba(X_new)
print(Y_proba)
sum=Y_proba[0,0]+Y_proba[0,1]+Y_proba[0,2]
print(sum)