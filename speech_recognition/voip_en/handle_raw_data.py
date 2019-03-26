import numpy as np
from scipy.io.wavfile import read, write
import os
from common import OneHot

n_sec_wav = 8
rate_wav = 16000

# 代表空单词
SPACE_TJL='SPACE_TJL'

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

def map_y_onehot(y_list,y_fpath,y_onehot_fpath):
    '''

    :param y_list: 一条音频文件对应的单词列表
    :param y_fpath: 已经读取后的包含所有单词的y文件
    :param y_onehot_fpath: 已经读取后的包含所有单词的y的onehot编码文件
    :return:返回这条音频文件单词对应的onehot编码列表
    '''
    y_data=list(np.load(y_fpath))
    y_onehot_data=np.load(y_onehot_fpath)
    tmp_list=[]
    for y in y_list:
        index=y_data.index(y)
        tmp_list.append(y_onehot_data[index,:])
    return tmp_list


# 由于机器性能原因可能导致无法进行训练，有可能需要以时间换空间，不一次性的将x读进来
def get_x(x_path):
    '''
    :param x_path: 已经读取好了的x数据文件存放路径
    :return:
    '''
    return np.load(x_path)

def get_y(indexs,datapath,n):
    '''
    :param indexs: x对应的索引列表，也是y对应的索引列表（x与y是相同的文件名）
    :param datapath::TRN文件所在目录
    :param n::一条音频文件对应的单词数量，不足的，用空补充
    :return:这些索引对应x的单词，经过onehot编码后的矩阵
    '''
    fpath =datapath
    whole_file_list = get_file_names(fpath)
    trn_file_list = []
    y_batch=[]
    for file in whole_file_list:
        if (file.find('TRN') >= 0):
            trn_file_list.append(file)
    for index in indexs:
        trn_file=trn_file_list[index]
        y_tmp=read_trn(trn_file)
        for i in range(n-len(y_tmp)):
            y_tmp.append(SPACE_TJL)
        y_onehot_tmp=map_y_onehot(y_tmp,
                                  y_fpath='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y1_with_SPACE_TJL.npy',
                                  y_onehot_fpath='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y1_with_SPACE_TJL_onehot.npy')
        y_onehot_tmp_mat=np.mat(y_onehot_tmp)
        y_batch.append(y_onehot_tmp_mat)
    return y_batch

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

    '''
    y1=np.load('D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y1.npy')
    # y2=np.load('D:\\学习笔记\\ai\\dataSets\\data_voip_en\\y2.npy')
    y1=list(y1)
    # y2=list(y2)
    y1.append(SPACE_TJL)
    y=y1
    np.save('y1_with_SPACE_TJL',y)
    y=OneHot.sklearn_one_hot(y)
    print('y_onehot', y)
    print(type(y))
    print(y['onehot_encoded'].shape)
    np.save('y1-y1_with_SPACE_TJL_onehot',y['onehot_encoded'])
    '''

    indexs=[1,2,3]
    data_set_dir = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData'
    max_line_char_num=20
    res=get_y(indexs=indexs, datapath=data_set_dir, n=max_line_char_num)

