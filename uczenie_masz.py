# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:36:32 2022

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



o = cv2.imread("PA_7_ref.png")
lo, X = ekstrakcja_cech(o)
y = np.ravel(ekstrakcja_klas(o))
yy = keras.utils.to_categorical(y,6)


lista = [i for i in range(18)]
wszystkie = pokazywanie_obiektow(o, lista)

rozmiar = 25
obrazy = np.zeros([len(lo),rozmiar,rozmiar,3])
print(obrazy[1])
for i in range(len(lo)):
    obrazy[i,:,:,0] = cv2.resize(lo[i],(rozmiar,rozmiar))
print(obrazy.shape)
rozmiar_ob = (rozmiar,rozmiar,1)

print(y)
pokazywanie_obiektow(o, [1])
b = cv2.inRange(o,(1,1,1),(255,255,255))
# etykietowanie i ekstrakcja cech
lol = regionprops(label(b))
print(lol[1]['Coordinates'][0])

#model neuronowy





