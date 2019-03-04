from PIL import Image
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
# from skimage import io,data
from sklearn.externals import joblib
import matplotlib.pyplot as plt

img=Image.open('lena.jpg')
img_data=np.array(img)
print(img_data.shape)
# img.show()
# print(img_data)
img_data_train=img_data
img_data_train=img_data_train.reshape(-1,3)
# clus=KMeans(n_clusters=256, init='k-means++')
clus=MiniBatchKMeans(n_clusters=480, init='k-means++',batch_size=1000)
clus.fit(img_data_train)
# joblib.dump(clus,'clus_img.pkl')
# clus=joblib.load('clus_img.pkl')
# print(clus.cluster_centers_)
# print(clus.labels_)
img_data_labels=clus.predict(img_data_train)
img_data_predict=clus.cluster_centers_[img_data_labels]
img_data_predict=np.array(img_data_predict,dtype=np.uint8)
print(img_data_predict)
img_data_predict=img_data_predict.reshape(512, 512, 3)

# img_source=Image.fromarray(img_data)
# img_source.show()
# img_predict=Image.fromarray(img_data_predict)
# img_predict.show()
# plt.imshow(img_data)
plt.imshow(img_data_predict)
plt.show()