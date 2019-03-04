import numpy as np
import matplotlib.pyplot as plt

X=np.random.rand(100,1)
X_b=np.c_[np.ones((100,1)),X]
Y=5+7*X+np.random.randn(100,1)


flag=False
n_epochs=2000
m=100
learn_rate=0.1
t0,t1=10,50
mini_bat=5
def learning_schedule(t):
    return t0/(t1+t)

theta=np.random.randn(2,1)

for epoch in range(n_epochs):
    for j in range(m):
        #random line
        random_index=np.random.randint(m-mini_bat)
        xi=X_b[random_index:random_index+mini_bat]
        yi=Y[random_index:random_index+mini_bat]
        gradient=xi.T.dot(xi.dot(theta)-yi)/mini_bat

        learn_rate=j+epoch*m
        learn_rate=learning_schedule(learn_rate)
        theta=theta-gradient*learn_rate

print(theta)
print(flag)

X_test=np.array([[0],[1]])
X_test_b=np.c_[np.ones((2,1)),X_test]
Y_test=X_test_b.dot(theta)
plt.plot(X,Y,'b.')
plt.plot(X_test,Y_test,'r-')
plt.axis([0,1,0,15])
plt.show()