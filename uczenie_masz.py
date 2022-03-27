# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:36:32 2022

@author: Filip
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


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





model = Sequential()

model = Sequential()
model.add(Dense(180, input_dim=18, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(18, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.save('model')





