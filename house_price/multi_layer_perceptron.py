from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data=pd.read_csv('house_price.csv',encoding = "utf-8")
data=np.array(data)
X=data[:,:3]
y=data[:,3:4]
scaler.fit(X)
X = scaler.transform(X)

mlpr=MLPRegressor(hidden_layer_sizes=(20,20,20),max_iter=1000,tol=0.01)
mlpr.fit(X,y)
X_test=data[:20,:3]
y_test=y[:20,:]
y_predict=mlpr.predict(X_test)
print(y_predict)
# print('coef.shape',[coef.shape for coef in mlpr.coefs_])
# print([coef for coef in mlpr.coefs_])





