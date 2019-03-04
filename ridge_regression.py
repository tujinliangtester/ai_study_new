import numpy
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
X=numpy.random.rand(100,1)
Y=3+4*X+numpy.random.randn(100,1)

plt.plot(X,Y,'b.')


X_test=numpy.array([[1.5],[0.5],[0]])

# Ridge
ridge_model=Ridge(alpha=0.1,solver='sag')
ridge_model.fit(X,Y)
print(ridge_model.intercept_)
print(ridge_model.coef_)
Y_predict=ridge_model.predict(X_test)
print(ridge_model.predict(X_test))
plt.plot(X_test,Y_predict,'r-')

print('SGDG')
SGDRegressor_model=SGDRegressor(penalty='l1',alpha=0.1,tol=1e-3)
SGDRegressor_model.fit(X,Y.ravel())
print(SGDRegressor_model.intercept_)
print(SGDRegressor_model.coef_)
# X_test2=numpy.array([[1.5]])
Y_predict2=SGDRegressor_model.predict(X_test)
print(SGDRegressor_model.predict(X_test))

plt.plot(X_test,Y_predict2,'g-')
plt.axis([0,2,0,10])
# plt.show()

# for two vars is ok!
#have to check the Y_2 is must made by 2D form X_2,the ':' is expected!!!
print('for two vars')
X_2=numpy.random.rand(10,2)
Y_2=3+4*X_2[:,0:1]+5*X_2[:,1:]+numpy.random.randn(10,1)
ridge_model=Ridge(solver='sag',max_iter=1000)
ridge_model.fit(X_2,Y_2)
print(222222222222222222222222222)
print(ridge_model.intercept_)
print(333)
print(ridge_model.coef_)
X_test=numpy.array([[2]])
X_test=numpy.c_[numpy.ones((1,1)),X_test]
# print()
print(ridge_model.predict(X_test))