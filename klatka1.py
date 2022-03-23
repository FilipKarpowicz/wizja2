# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:21:28 2022

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
#from projekt_v1 import pokaz,polob,ekstrakcja_cech,ekstrakcja_klas,zamiana_bgr2hsv,zamiana_hsv2bgr,pokazywanie_obiektow


import time



wideo = cv2.VideoCapture('PA_7.avi')



while(True):
    ret, ramka = wideo.read()
    
    
    
    cv2.imshow('oryginalny', ramka)
    
    
    
    cv2.imwrite('klatak.png',ramka)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break
 
wideo.release()       
cv2.destroyAllWindows()