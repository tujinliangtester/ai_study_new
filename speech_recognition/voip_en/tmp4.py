import numpy as np


y=np.load('D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y1_with_SPACE_TJL_onehot.npy')

print(y.shape)
for i in range(y.shape[0]):
    tmp=y[i,:]
    if(np.argmax(tmp)==151):
        print(i)
