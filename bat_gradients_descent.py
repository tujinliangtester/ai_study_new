import numpy as np

X=3*np.random.rand(100,1)
Y=5+7*X+np.random.randn(100,1)

X_b=np.c_[np.ones((100,1)),X]

theta=np.random.rand(2,1)
rate=0.1
n_iter=100000
m=100

for i in range(n_iter):

    gradients=1/m*X_b.T.dot(X_b.dot(theta)-Y)
    theta=theta-gradients*rate
print(theta)