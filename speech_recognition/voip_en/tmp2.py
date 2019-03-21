import numpy as np
from scipy.io.wavfile import read, write
import os
from common import OneHot

n_sec_wav = 8
rate_wav = 16000


def read_wavs(fname_list):
    x = np.mat(np.zeros(shape=(1, n_sec_wav * rate_wav)))
    shape_x = n_sec_wav * rate_wav
    for i, fname in enumerate(fname_list):
        rate, data = read(fname)
        shape1 = data.shape
        if (shape1[0] < shape_x):
            data_0 = np.ones(shape=(shape_x - shape1[0]))
            # data.extend(data_0)
            data = np.append(data, data_0)
        data = np.mat(data)
        x = np.vstack((x, data))
        # x.append(data)
        if (i % 100 == 0):
            print(i, x[i:i + 1, :])
    return x[1:]


def read_wav():
    rate_list = []
    data_shape_list = []
    s = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\jurcic-001-120912_124317_0001940_0002325.wav'
    fname_list = [s]
    data = ''
    for fname in fname_list:
        rate, data = read(fname)
        rate_list.append(rate)
        data_shape_list.append(data.shape)
    print(data)
    print(data.shape)
    '''
    rate    16000
    n_sec_wav   6s
    '''


get_file_names_tmp_list = []


def get_file_names(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_file_names(fi_d)
        else:
            get_file_names_tmp_list.append(fi_d)
    return get_file_names_tmp_list


def read_trn(fpath):
    with open(fpath) as f:
        res = f.read()
        return res.split(' ')

def load_data(one_hot=True,filepath='D:\\学习笔记\\ai\\dataSets\\number-wav-data\\'):
    # 已经调用方法将音频文件转换成矩阵文件
    data = np.load(filepath )
    if(one_hot==True):
        data=OneHot.sklearn_one_hot(data)['onehot_encoded']
    return data

if __name__ == '__main__':

    '''
    fpath = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\'
    whole_file_list = get_file_names(fpath)
    wav_file_list = []
    trn_file_list = []
    for file in whole_file_list:
        if (file.find('TRN') >= 0):
            trn_file_list.append(file)
        else:
            wav_file_list.append(file)
    # print(len(wav_file_list))  39436
    '''

    '''
    end=0
    for i in range(1,41):
        start=end
        end=i*1000
        if(end>39436):
            end=39436
        x=read_wavs(wav_file_list[start:end])
        print(x[0])
        print(x.shape)
        outputx_name='x'+str(i)
        np.save(outputx_name,x)
        x=0
    '''

    '''
    end = 0
    for i in range(1, 3):
        y_word = []
        start = end
        end = i * 1000
        for file in trn_file_list[start:end]:
            y_word=y_word+read_trn(file)
        output_y_name = 'y' + str(i)
        np.save(output_y_name, y_word)
        print(len(y_word))
        print(y_word[0])
    '''
    y1=np.load('D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y1.npy')
    y2=np.load('D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y2.npy')
    y1=list(y1)
    y2=list(y2)
    y=y1+y2
    # np.save('y1-2',y)
    y=OneHot.sklearn_one_hot(y)
    print('y_onehot', y)
    print(type(y))
    print(y['onehot_encoded'].shape)
    # np.save('y1-2_onehot_encoded',y['onehot_encoded'])
