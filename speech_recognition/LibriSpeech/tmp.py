import gzip

import soundfile as sf
import sys, os
import re

'''
data_path='D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\84\\121123\\'+'84-121123-0002.flac'
data,rate=sf.read(data_path)

print('data:',data)
print(type(data))
print(max(data))
print(min(data))
print(data.shape)
print('rate:',rate)

'''

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


filepath = 'D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\'
fname_list = get_file_names(filepath)
fname_list_trans = []
for i in fname_list:
    if (i.find('trans') >= 0):
        fname_list_trans.append(i)
print(fname_list_trans)


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


# ids,char_list=read_file('D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\1272\\128104\\1272-128104.trans.txt')
# print('ids:',ids)
# print('char_list:',char_list)

# fname_list_trans = [
#     'D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\1272\\128104\\1272-128104.trans.txt',
#     'D:\\学习笔记\\ai\\dataSets\\dev-clean\\LibriSpeech\\dev-clean\\1272\\135031\\1272-135031.trans.txt']
ids = {}
char_list = []
end = 0
for fpath in fname_list_trans:
    tmp_ids, tmp_char_num, tmp_char_list = read_file(end=end, fpath=fpath)
    end = tmp_char_num
    ids = dict(ids, **tmp_ids)
    char_list = char_list + tmp_char_list

print('ids:', ids)
print('char_list:', char_list)
print(len(char_list))