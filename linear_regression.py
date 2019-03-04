import numpy as np
import matplotlib.pyplot as plt

X=2*np.random.rand(100,1)
Y=4+3*X+np.random.randn(100,1)

X_b=np.c_[np.ones((100,1)),X]
# print(X_b)

theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print('theta')
print(theta)

X_test=np.array([[0],[1],[4]])
X_test_b=np.c_[np.ones((3,1)),X_test]

Y_predict=X_test_b.dot(theta)
print(Y_predict)

plt.plot(X_test,Y_predict,'r-')
plt.plot(X,Y,'b.')
plt.axis([0,2,0,15])
plt.show()
