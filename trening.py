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


o = cv2.imread("PA_7_ref.png")
lo, X = ekstrakcja_cech(o)
y = np.ravel(ekstrakcja_klas(o))
yy = keras.utils.to_categorical(y,18)

xx = np.zeros(18)
lista = [i for i in range(18)]
wszystkie = pokazywanie_obiektow(o, lista)
lista2 = []
for i in range(18):
    lista2.append(wszystkie[i])
x = np.array(lista2)





yy = keras.utils.to_categorical(y,18)

model = keras.models.load_model('model')


trening = model.fit(X, yy, epochs=550, batch_size=15)

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

y_pred = model.predict(X)
y_pred_max = np.amax(y_pred,1)
y_pred_id = np.array([np.argwhere(y_pred==maxval).flatten()[1] for maxval in y_pred_max])
y_pred_id
y
