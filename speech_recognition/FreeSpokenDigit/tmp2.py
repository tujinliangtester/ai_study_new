from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
# print("=========All Variables==========")
# print_tensors_in_checkpoint_file("tmp/model.ckpt", tensor_name=None, all_tensors=True, all_tensor_names=True)

from tensorflow.contrib import legacy_seq2seq

# tmp_attention_decoder=legacy_seq2seq.one2many_rnn_seq2seq()


import samplerate
import numpy as np
# from . import  LoadData
# from speech_recognition.FreeSpokenDigit import LoadData
from scipy.io.wavfile import read, write


def read_wavs(fname_list):
    # x = np.mat(np.zeros(shape=(1, n_sec_wav*rate_wav)))
    # shape_x=n_sec_wav*rate_wav
    # y = np.mat(np.zeros(shape=(1)))
    data=None
    label=None

    for i,fname in enumerate(fname_list):
        label = fname.split('/')[-1].split('_')[0].split('\\')[-1]
        rate, data = read(fname)
        data = data[:, 0]
        '''
        注意，这里进行了一些处理，由于音频文件的时间长度不一，采样率都是8000，则数据大小不一
        先按照每个音频文件3秒（最大的2.3秒，足够），即3*8000的长度来考虑，不足的用1来补充
        '''
        # shape1 = data.shape
        # if(shape1[0]<shape_x):
        #     data_0=np.ones(shape=(shape_x-shape1[0]))
        #     # data.extend(data_0)
        #     data=np.append(data,data_0)
        # data = np.mat(data)

        # x = np.vstack((x, data))
        # x.append(data)
        # if(i%100==0):
        #     print(i,x[i:i+1,:])
    return data, label

fname_list=['D:\\学习笔记\\ai\\dataSets\\number-wav-recordings\\1_tujinliang_0.wav']
# fname_list.append('D:\\学习笔记\\ai\\dataSets\\number-wav-recordings\\0_jackson_0.wav')

x,y=read_wavs(fname_list)
print('x shape',x.shape)
output_sample_rate=8000
input_sample_rate=44100
ratio=output_sample_rate / input_sample_rate
output=samplerate.resample(input_data=x,ratio=ratio)
print('output',output)
print('output shape',output.shape)
