import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
rad='D:\\ML\\data_set\\music\\1.mp3'
sample_rat,X=wavfile.read(rad)
fft_features=abs(fft(X)[:1000])
sad=np.save('.\\1.fft',fft_features)


Y=np.array([[1]])

