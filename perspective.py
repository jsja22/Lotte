import os
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
import cv2
from tqdm import tqdm

rotXdeg = 90
rotYdeg = 90
rotZdeg = 90
f = 250
dist = 300

def onRotXChange(val):
    global rotXdeg
    rotXdeg = val
def onRotYChange(val):
    global rotYdeg
    rotYdeg = val
def onRotZChange(val):
    global rotZdeg
    rotZdeg = val
def onFchange(val):
    global f
    f=val
def onDistChange(val):
    global dist
    dist=val

for i in tqdm(range(1000)):
    for j in range(24,48):
        original = cv2.imread(f'C:/data/LPD_competition/train/{i}/{j}.jpg')


        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

        # 커널 적용 
        new_img = np.ndarray(shape=original.shape,dtype=original.dtype)
        h , w = original.shape[:2]

        print(h, w)
        rotX = (25)*np.pi/180
        rotY = 0
        rotZ = 0

        A1= np.matrix([[1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0   ],
                    [0, 0, 1   ]])

        RX = np.matrix([[1,           0,            0, 0],
                        [0,np.cos(rotX),-np.sin(rotX), 0],
                        [0,np.sin(rotX),np.cos(rotX) , 0],
                        [0,           0,            0, 1]])

        RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                        [            0, 1,            0, 0],
                        [ -np.sin(rotY), 0, np.cos(rotY), 0],
                        [            0, 0,            0, 1]])

        RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                        [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                        [            0,            0, 1, 0],
                        [            0,            0, 0, 1]])

        R = RX * RY * RZ

        T = np.matrix([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,dist],
                    [0,0,0,1]])

        A2= np.matrix([[f, 0, w/2,0],
                    [0, f, h/2,0],
                    [0, 0,   1,0]])

        H = A2 * (T * (R * A1))

        cv2.warpPerspective(original, H, (w, h), new_img, cv2.INTER_CUBIC)
        new_img = cv2.resize(new_img,(300,560))
        new_img = new_img[0:356, 0:256].copy()
        new_img = cv2.resize(new_img,(256,256))
        new_img = new_img[18:250, 30:256].copy()

        cv2.imwrite('C:/data/LPD_competition/train/{}/{}.jpg'.format(i,j+24), new_img)


