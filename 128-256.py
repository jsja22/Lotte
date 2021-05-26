from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Activation,LeakyReLU,UpSampling2D,Input,Dense,Reshape,Flatten,Conv2DTranspose,ReLU,concatenate,ZeroPadding2D,UpSampling2D
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
x = np.load('C:/data/LPD_competition/npy/LPD__x_128.npy',allow_pickle=True)

print(x.shape)  #(72000, 128, 128, 3)

X=[]
for i in tqdm(range(72000)):
    img = x[i,:,:,:]
    img2 = cv2.resize(img, dsize=(256,256),interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
    img3 = np.asarray(img2)
    X.append(img3)

np.save("C:/data/LPD_competition/npy/LPD_x_128_1.npy", arr=X)

x = np.load("C:/data/LPD_competition/npy/LPD_x_128_1.npy",allow_pickle=True)

print(x.shape)
