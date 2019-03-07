from scipy.io.wavfile import read, write
import sys, os
import numpy as np
from common import OneHot

#每个音频文件规定长度
n_sec_wav=3
rate_wav=8000

def get_file_names(filepath):
    files = os.listdir(filepath)
    tmp_list = []
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_file_names(fi_d)
        else:
            tmp_list.append(fi_d)
            print(os.path.join(filepath, fi_d))
    return tmp_list


def read_wavs(fname_list):
    x = np.mat(np.zeros(shape=(1, n_sec_wav*rate_wav)))
    shape_x=n_sec_wav*rate_wav
    y = np.mat(np.zeros(shape=(1)))
    for i,fname in enumerate(fname_list):
        label = fname.split('/')[-1].split('_')[0].split('\\')[-1]
        y = np.vstack((y, label))
        # y.append(label)
        # here  fname is of full directory
        rate, data = read(fname)
        '''
        注意，这里进行了一些处理，由于音频文件的时间长度不一，采样率都是8000，则数据大小不一
        先按照每个音频文件3秒（最大的2.3秒，足够），即3*8000的长度来考虑，不足的用1来补充
        '''
        shape1 = data.shape
        if(shape1[0]<shape_x):
            data_0=np.ones(shape=(shape_x-shape1[0]))
            # data.extend(data_0)
            data=np.append(data,data_0)
        data = np.mat(data)

        x = np.vstack((x, data))
        # x.append(data)
        if(i%100==0):
            print(i,x[i:i+1,:])
    return x, y


def read_wav(fname_list):
    rate_list = []
    data_shape_list = []
    for fname in fname_list:
        rate, data = read(fname)
        rate_list.append(rate)
        data_shape_list.append(data.shape)
    print('max(rate_list)', max(rate_list))
    print('min(rate_list)', min(rate_list))
    print('max(data_shape_list)', max(data_shape_list))
    print('min(data_shape_list)', min(data_shape_list))
    # print('rate:',rate,'data:',data,'data shape:',data.shape)
    '''
    max(rate_list) 8000
    min(rate_list) 8000
    max(data_shape_list) (18262,)
    min(data_shape_list) (1148,)
    '''
def load_data(one_hot):
    # 已经调用方法将音频文件转换成矩阵文件
    filepath = 'D:\\学习笔记\\ai\\dataSets\\number-wav-data\\'
    x2 = np.load(filepath + 'x.npy')
    y2 = np.load(filepath + 'y.npy')
    if(one_hot==True):
        y2=OneHot.sklearn_one_hot(y2)['onehot_encoded']
    return x2,y2

def shuffle_data():
    filepath = 'D:\\学习笔记\\ai\\dataSets\\number-wav-data\\'
    x2 = np.load(filepath + 'x.npy')
    y2 = np.load(filepath + 'y.npy')
    x2=np.hstack((x2,y2))
    np.random.shuffle(x2)
    x_shuffled=x2[:,0:24000]
    y_shuffled=x2[:,-1]
    np.save('x_shuffled',x_shuffled)
    np.save('y_shuffled',y_shuffled)
    print('shuffle_data done')

if __name__ == '__main__':
    '''

    filepath='D:\\学习笔记\\ai\\dataSets\\number-wav-recordings\\'
    fname_list = get_file_names(filepath)
    np.random.shuffle(fname_list)
    # tmp_list = fname_list  # [:2]
    # read_wav(tmp_list)
    x,y=read_wavs(fname_list=fname_list)
    np.save('x',x[1:,:])
    np.save('y',y[1:,:])
    print('done')
    '''

    filepath='D:\\学习笔记\\ai\\dataSets\\number-wav-data\\'
    x2=np.load(filepath+'x.npy')
    y2=np.load(filepath+'y.npy')
    print(y2)
    # OneHot.sklearn_one_hot(y2[1:,:])

