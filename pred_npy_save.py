import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input
from tqdm import tqdm

img1=[]
for i in tqdm(range(0,72000)):
        filepath=f'C:/data/LPD_competition/pred/test/{i}.jpg'
        image2=Image.open(filepath)
        image2 = image2.convert('RGB')
        image2 = image2.resize((224,224))
        image_data2=asarray(image2)
        img1.append(image_data2)    

np.save('C:/data/LPD_competition/npy/Final_x_pred.npy', arr=img1)

x_pred = np.load('C:/data/LPD_competition/npy/Final_x_pred.npy',allow_pickle=True)

print(x_pred.shape) 