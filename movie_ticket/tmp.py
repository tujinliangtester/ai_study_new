import numpy as np
from  common import    OneHot

fpath='E:\\tjl_ai\\dataSet\\movie_ticket'


def save_x():
    with open(fpath+'\\x.txt') as f:
        fline_list=f.readlines()

    x=np.mat(np.zeros(shape=(2)))
    for line in fline_list:
        line_splited=line.split('\n')[0].split('\t')
        if(line_splited==['']):break
        mat_line_splited=np.mat(line_splited)
        x=np.vstack((x,mat_line_splited))
    print(x.shape)

    x_tmp=x[1:,]
    np.save('x',x_tmp)

def save_y():
    with open(fpath+'\\y.txt') as f:
        fline_list=f.readlines()

    y=np.mat(np.zeros(shape=(1)))
    for line in fline_list:
        line_splited=line.split('\n')[0].split('\t')
        if(line_splited==['']):break
        mat_line_splited=np.mat(line_splited)
        y=np.vstack((y,mat_line_splited))
    print(y.shape)
    y_tmp=y[1:,]
    np.save('y',y_tmp)

def save_y_onehot():
    y=np.load(fpath+'\\y.npy')
    y_onehot=OneHot.sklearn_one_hot(y)['onehot_encoded']
    print(y_onehot.shape)
    np.save('y_onehot',y_onehot)

def read_data(fname):
    fpath_tmp=fpath+'\\'+fname
    x_mat=np.load(fpath_tmp)
    return x_mat



if __name__=='__main__':
    y_ = read_data('y.npy')
    print(y_)
