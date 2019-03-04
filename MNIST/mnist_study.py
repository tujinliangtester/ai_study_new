import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score,roc_curve,roc_auc_score
from sklearn.externals import joblib

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

if __name__=='__main__':

    images, labels=load_mnist('.',kind='train')
    X_train,y_train=images, labels

    images, labels = load_mnist('.', kind='t10k')
    X_test, y_test = images, labels
    y_train = (y_train == 5)
    y_test = (y_train == 5)
    '''
    # x=X_train[3600].reshape(28,28)
    # plt.imshow(x)
    # plt.axis('off')
    # plt.show()
    
    print(y_train)
    

    lclf=LogisticRegression(max_iter=100,solver='sag')
    lclf.fit(X_train,y_train)
    joblib.dump(lclf,'lclf.pkl')
    print('fit is done')
    '''
    lclf=joblib.load('lclf.pkl')
    y_valid=lclf.predict(X_train)

    acc=accuracy_score(y_train,y_valid)
    print(acc)

    precision=precision_score(y_train,y_valid)
    print(precision)

    recall=recall_score(y_train,y_valid)
    print(recall)

    f1=f1_score(y_train,y_valid)
    print(f1)

    fpr, tpr, thresholds = roc_curve(y_train, y_valid)
    print(fpr,tpr,thresholds)

    plt.plot(fpr,tpr,'b.')
    plt.show()