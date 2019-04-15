import os
import sys
import platform
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
from scipy.io.wavfile import read, write


def time2freq(sigs,sample_rate):
    freqs = nf.fftfreq(len(sigs),d=1/sample_rate)
    ffts = nf.fft(sigs)
    print('freqs:',freqs,freqs.shape)
    print(ffts)
    amps = np.abs(ffts)
    print(amps)
    return freqs,ffts,amps

def read_wav():
    rate_list = []
    data_shape_list = []
    s = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\jurcic-001-120912_124317_0001940_0002325.wav'
    fname_list = [s]
    data = ''
    rate=''
    for fname in fname_list:
        rate, data = read(fname)
        rate_list.append(rate)
        data_shape_list.append(data.shape)
    print(data)
    print(data.shape)
    return rate, data


if __name__=='__main__':
    rate, data=read_wav()
    freqs, ffts, amps=time2freq(data,rate)