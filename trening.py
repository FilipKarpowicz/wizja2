# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:21:06 2022

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


lo = []
X = []
y = []
for i in range(0,986):
    try:
        o = cv2.imread(f"data1/klatka{i}.png")
        lo1, X1 = ekstrakcja_cech(o)
        y1 = np.ravel(ekstrakcja_klas(o))
        lo.append(lo1[0])
        X.append(X1[0])
        y.append(y1[0])
    except:
        print('blad probka nr {i} nie jest prawidlowa!')
  
lo = np.array(lo)
X = np.array(X)
y = np.array(y)










yy = keras.utils.to_categorical(y,18)

model = keras.models.load_model('model')


trening = model.fit(X, yy, epochs=550, batch_size=30)

_, dokladnosc = model.evaluate(X, yy)
print('Dokładność: %.2f' % (dokladnosc*100))

blad = trening.history['loss']
dokladnosc = trening.history['accuracy']
plt.plot(blad)
plt.plot(dokladnosc)
plt.plot(np.ones([1,len(blad)]).ravel())
plt.legend(['błąd', 'dokładność','1'])
plt.show()

model.save('model')


