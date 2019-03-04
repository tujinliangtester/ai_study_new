from sklearn import datasets
from sklearn.linear_model import LogisticRegression
'''
找不到具体的方法？是整合到逻辑回归中了吗？记得之前好像是遇到过的，但是查找又没有找到
明天需要复习一下softmax的网络拓扑、激活函数或者叫归一化函数（其实就是交叉熵）、尝试理解推导损失函数及偏导函数
用SGD的思路求解最优解的过程等。
当然，也可能可以从其他框架，如tensflow等尝试softmax

注意，损失函数并不一定就是按照某种理论（如交叉熵）推导出来的公式，可以在不影响单调性和各元素间相互比例的情况下，
对损失函数进行变形，使得其函数本身或对theta求偏导时更简单

'''
iris=datasets.load_iris()
X=iris['data']
Y=iris['target']
