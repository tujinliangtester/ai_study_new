#思想很重要，就是将高阶看做是新的维度，计算后当做一阶线性来处理即可。
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

X=10*np.random.rand(100,1)-5
Y=1+2*X+3*X**2+2*np.random.randn(100,1)
plt.plot(X,Y,'b.')
d={1:'g.',2:'r.',10:'y.'}
for i in d:
    poly_features=PolynomialFeatures(degree=i,include_bias=False)
    X_poly=poly_features.fit_transform(X)
    ridge_model=Ridge(alpha=1,solver='sag')
    ridge_model.fit(X_poly,Y)
    Y_b=ridge_model.predict(X_poly)
    plt.plot(X,Y_b,d[i])
plt.show()