import numpy as np
from scipy.io.wavfile import read, write
import os
from common import OneHot

n_sec_wav = 8
rate_wav = 16000

# 代表空单词
SPACE_TJL = 'SPACE_TJL'

# 单词数量
words_num = 0

# 音频长度
wav_lenth = 0

# 截取长度，毫秒*rate
split_lenth = 1 * rate_wav

# 步长
step_lenth = int(split_lenth / 2)

# 平均一秒钟对应的单词数，取每秒钟2个单词
words_num_per_min = 1.49
words_num_per_split = int(2 * split_lenth / rate_wav)

# 针对trn大小写的兼容性问题
trn_name = 'trn'


def read_wavs(fname_list):
    x = np.mat(np.zeros(shape=(1, n_sec_wav * rate_wav)))
    shape_x = n_sec_wav * rate_wav
    for i, fname in enumerate(fname_list):
        rate, data = read(fname)
        shape1 = data.shape
        if (shape1[0] < shape_x):
            # data_0 = np.ones(shape=(shape_x - shape1[0]))
            data_0 = np.zeros(shape=(shape_x - shape1[0]))
            # data.extend(data_0)
            data = np.append(data, data_0)
        data = np.mat(data)
        x = np.vstack((x, data))
        # x.append(data)
        if (i % 100 == 0):
            print(i, x[i:i + 1, :])
    return x[1:]


def read_wavs_as_array(fname_list):
    x = None
    for i, fname in enumerate(fname_list):
        rate, data = read(fname)
        x = np.array(data)
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


def read_wav_2(s):
    rate_list = []
    data_shape_list = []
    fname_list = [s]
    data = ''
    for fname in fname_list:
        rate, data = read(fname)
        rate_list.append(rate)
        data_shape_list.append(data.shape)
    '''
        rate    16000
        n_sec_wav   6s
    '''
    return data.shape


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


def load_data(one_hot=True, filepath='D:\\学习笔记\\ai\\dataSets\\number-wav-data\\'):
    # 已经调用方法将音频文件转换成矩阵文件
    data = np.load(filepath)
    if (one_hot == True):
        data = OneHot.sklearn_one_hot(data)['onehot_encoded']
    return data


def map_y_onehot(y_list, y_fpath, y_onehot_fpath):
    '''

    :param y_list: 一条音频文件对应的单词列表
    :param y_fpath: 已经读取后的包含所有单词的y文件
    :param y_onehot_fpath: 已经读取后的包含所有单词的y的onehot编码文件
    :return:返回这条音频文件单词对应的onehot编码列表
    '''
    y_data = list(np.load(y_fpath))
    y_onehot_data = np.load(y_onehot_fpath)
    tmp_list = []
    for y in y_list:
        index = y_data.index(y)
        tmp_list.append(y_onehot_data[index, :])
    return tmp_list


# 由于机器性能原因可能导致无法进行训练，有可能需要以时间换空间，不一次性的将x读进来
def get_x(x_path):
    '''
    :param x_path: 已经读取好了的x数据文件存放路径
    :return:
    '''
    return np.load(x_path)


def get_y(indexs):
    '''
    :param indexs: x对应的索引列表，也是y对应的索引列表（x与y是相同的文件名）
    :return:这些索引对应x的单词，经过onehot编码后的矩阵
    '''
    y_splited_with_space_tjl=np.load('E:\\tjl_ai\\dataSet\\tmpOut\\'+'y_splited.npy')
    y_list_with_space_tjl=list(np.load('E:\\tjl_ai\\dataSet\\tmpOut\\'+'y_with_SPACE_TJL.npy'))
    y_with_SPACE_TJL_onehot=np.load('E:\\tjl_ai\\dataSet\\tmpOut\\'+'y_with_SPACE_TJL_onehot.npy')

    y_batch = []

    for index in indexs:
        y_tmp=[]
        y_line = list(y_splited_with_space_tjl[index,])
        for word_tmp in y_line:
            i=y_list_with_space_tjl.index(word_tmp)
            y_onehot_tmp_line=y_with_SPACE_TJL_onehot[i]
            y_tmp.append(y_onehot_tmp_line)
        y_tmp=np.mat(y_tmp)
        y_batch.append(y_tmp)
    return y_batch


def get_words_wav_num_whole(whole_file_list):
    flag = True
    words_num_whole = 0
    wav_lenth_whole = 0
    for i in range(len(whole_file_list)):
        if (flag == False):
            flag = True
            continue
        file = whole_file_list[i]
        if (file.find(trn_name) >= 0):
            content = read_trn(file)
            if (len(content) == 0):
                flag = False
                continue
            else:
                words_num_whole += len(content)
        else:
            tmp_length = read_wav_2(file)
            wav_lenth_whole += tmp_length[0]
    print('words_num_whole:', words_num_whole)
    print('wav_lenth_whole:', wav_lenth_whole)


def split_wav_fun(x, file_data, split_lenth):
    '''
    切分音频文件，返回叠加后的x矩阵及切分个数
    注意：这里要求file data的长度大于等于split_lenth
    :param x:目标x的矩阵
    :param file_data:当前文件读取后的数据
    :param split_lenth:切分音频长度
    :return:
    x:返回叠加后的x矩阵
    i:返回总共切分成了多少个
    '''
    data = file_data
    i = 0
    while (len(data) > split_lenth):
        res_tmp = data[:split_lenth]
        res_tmp = np.mat(res_tmp)
        x = np.vstack((x, res_tmp))
        i += 1
        # 步进
        data = data[step_lenth:]
    if (len(data) == split_lenth):
        res_tmp = data[:split_lenth]
        res_tmp = np.mat(res_tmp)

        x = np.vstack((x, res_tmp))
        i += 1
    if (len(data) < split_lenth):
        # 如果音频文件不足，则取前面的
        res_tmp = file_data[-split_lenth:]
        res_tmp = np.mat(res_tmp)
        x = np.vstack((x, res_tmp))
        i += 1
    return x, i


def split_trn_fun(y, words_list, split_num):
    '''
    根据音频文件切分的个数，对对应的trn文件单词进行切分
    :param y: 单词切分后的矩阵
    :param words_list: trn文件对应的单词列表
    :param split_num: 对应音频文件切分的次数
    :return:y，叠加后的单词矩阵
    '''
    # 单词不足时，用SPACE_TJL补充
    while (len(words_list) < words_num_per_split):
        words_list.append(SPACE_TJL)
    for i in range(split_num):
        start = int(i * words_num_per_split / 2)
        end = start + words_num_per_split
        if (end + 1 > len(words_list)):
            res_tmp = words_list[-words_num_per_split:]
            res_tmp = np.mat(res_tmp)
            y = np.vstack((y, res_tmp))
        else:
            res_tmp = words_list[start:end]
            res_tmp = np.mat(res_tmp)
            y = np.vstack((y, res_tmp))
    return y


def split_wav_trn(whole_file_list):
    '''
    切分音频、trn文件
    :param whole_file_list: 音频、trn文件列表
    :return: 返回音频、单词切分后的矩阵
    '''
    x = np.mat(np.zeros(shape=(1, split_lenth)))
    y = np.mat(np.zeros(shape=(1, words_num_per_split)))
    j=0
    for file in whole_file_list:
        j+=1
        if(j%100==0):
            print(j)
        if (file.find(trn_name) < 0):
            _, wav_data = read(file)
            x, i = split_wav_fun(x, wav_data, split_lenth)
            trn_file = file + '.' + trn_name
            words_list = read_trn(trn_file)
            y = split_trn_fun(y, words_list, i)
    return x[1:, ], y[1:, ]

def save_word_list(trn_file_list):
    y_word = []
    for file in trn_file_list:
        y_word=y_word+read_trn(file)
    output_y_name = 'y_list'
    np.save(output_y_name, y_word)
    print(len(y_word))
    print(y_word[0])

def save_word_onehot(y_word_path):
    y1=np.load(y_word_path)
    y1=list(y1)
    y1.append(SPACE_TJL)
    y=y1
    np.save('y_with_SPACE_TJL',y)
    y=OneHot.sklearn_one_hot(y)
    print('y_onehot', y)
    print(type(y))
    print(y['onehot_encoded'].shape)
    np.save('y_with_SPACE_TJL_onehot',y['onehot_encoded'])

if __name__ == '__main__':
    # fpath = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\'
    fpath = 'E:\\tjl_ai\\dataSet\\tmp\\'
    whole_file_list = get_file_names(fpath)
    wav_file_list = []
    trn_file_list = []
    for file in whole_file_list:
        if (file.find(trn_name) >= 0):
            trn_file_list.append(file)
        else:
            wav_file_list.append(file)

    x, y = split_wav_trn(whole_file_list)
    print('x:', x)
    print('y:', y)
    np.save('x_splited', x)
    np.save('y_splited', y)

    save_word_list(trn_file_list)

    cur_path='E:\\tjl\\AI\\语音识别\\ai_study_new\\speech_recognition\\voip_en\\y_list.npy'
    save_word_onehot(cur_path)



