import os
import cv2
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드
caltech_dir =  'C:/data/LPD_competition/train_SR/'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)
image_w = 256
image_h = 256
pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("C:/data/LPD_competition/npy/Final_x.npy", arr=X)
np.save("C:/data/LPD_competition/npy/Final_y.npy", arr=y)

x = np.load("C:/data/LPD_competition/npy/Final_x.npy",allow_pickle=True)
y = np.load("C:/data/LPD_competition/npy/Final_y.npy",allow_pickle=True)

print(x.shape)
print(y.shape)


