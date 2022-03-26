# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:39:19 2022

@author: Filip
"""

import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import colorsys


from skimage.measure import label,regionprops

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import tree

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow import keras
from projekt_v1 import pokaz,polob,ekstrakcja_cech,ekstrakcja_klas,zamiana_bgr2hsv,zamiana_hsv2bgr,pokazywanie_obiektow


o = cv2.imread("data/test432.png")
lo, X = ekstrakcja_cech(o)
ka = ekstrakcja_klas(o)

model = keras.models.load_model('model')
ka = ka.tolist()
y_pred = model.predict(X)
y_pred_max = np.amax(y_pred,1)
y_pred_id = np.array([np.argwhere(y_pred==maxval).flatten()[1] for maxval in y_pred_max])
print(y_pred_id)
print(f'czy klasy sie zgadzaja {y_pred_id == ka} klasa modelu {y_pred_id} klasa z ekstrakcji {ka}')
plt.imshow(o)
for j in range(18):    
    x = np.where(y_pred_id == j)
    x = np.ndarray.tolist(x[0])
    obrazki = pokazywanie_obiektow(o,x)
    polob(obrazki,colmap='RGB',ile_k=3,osie=(True))