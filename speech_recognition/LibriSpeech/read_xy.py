
import soundfile as sf
import sys, os
from common import OneHot
import numpy as np

filepath = 'D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\'
n_sec_wav=30
rate_wav=16000

# 定义空单词
NULL_SPACE='NSP'

# 每个音频文件的单词数量，数据中最大是52，暂定60个单词
max_line_char_num=60

# 注意，这里的方法有点问题，在递归时会将tmp list重新置空，所以需要将tmp list 放在函数的外面
tmp_list = []
def get_file_names(filepath):
    files = os.listdir(filepath)

    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_file_names(fi_d)
        else:
            tmp_list.append(fi_d)
            # print(os.path.join(filepath, fi_d))
    return tmp_list

fname_list = get_file_names(filepath)

fname_list_flac=[]
for i in fname_list:
    if (i.find('flac') >= 0):
        fname_list_flac.append(i)

def read_flac2x(n_sec_wav, rate_wav, batch_fname_list_flac):
    shape_x=n_sec_wav*rate_wav
    x = np.mat(np.zeros(shape=(1, shape_x)))
    for fname_flac in batch_fname_list_flac:
        data,rate=sf.read(fname_flac)
        shape1=data.shape
        if(shape_x - shape1[0]>0):
            data_0 = np.ones(shape=(shape_x - shape1[0]))
            data=np.append(data,data_0)
        data = np.mat(data)
        x = np.vstack((x, data))
    return x[1:]

fname_list_trans = []
for i in fname_list:
    if (i.find('trans') >= 0):
        fname_list_trans.append(i)


def read_file(end, fpath):
    ids = {}
    char_list = []
    with open(fpath) as f:
        lines = f.readlines()
        print(lines[0])
        for line in lines:
            start = end
            line_detail = line.split(' ')
            for i in range(len(line_detail)):
                if (i == 0):
                    end = end + len(line_detail) - 1
                    ids[line_detail[i]] = [start, end]
                else:
                    tmp_char=line_detail[i]
                    tmp_char=tmp_char.replace('\n','')
                    char_list.append(tmp_char)
    char_num = end
    return ids, char_num, char_list

def read_trans2y():
    ids = {}
    char_list = []
    end = 0

    for fpath in fname_list_trans:
        tmp_ids, tmp_char_num, tmp_char_list = read_file(end=end, fpath=fpath)
        end = tmp_char_num
        ids = dict(ids, **tmp_ids)
        char_list = char_list + tmp_char_list

    # # 填充空单词，以便进行训练
    # char_list.append(NULL_SPACE)
    print('len(char_list)',len(char_list))
    encode_dic=OneHot.sklearn_one_hot(char_list)
    onehot_encoded=encode_dic['onehot_encoded']
    # print(onehot_encoded.shape)#(54402, 8333)

    # 填充空单词，以便进行训练
    col_zero=np.zeros(shape=(onehot_encoded.shape[0],1))
    onehot_encoded=np.hstack((onehot_encoded,col_zero))

    row_zero=np.zeros(shape=(1,onehot_encoded.shape[1]))
    row_zero[1,-1]=1
    print('row_zero:',row_zero)
    return onehot_encoded

# 根据x取不同的y，并且将不足60的部分补充成NPS对应的one hot 编码，应该是最后一个？
def gene_y():
    print(1)

if __name__=='__main__':
    # x=read_flac2x(n_sec_wav=n_sec_wav, rate_wav=rate_wav, batch_fname_list_flac=fname_list_flac[:5])
    # print(x)
    # print(x.shape)
    y=read_trans2y()
    print(y[-1,])
    print(y.shape)