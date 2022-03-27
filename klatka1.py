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


from time import sleep



wideo = cv2.VideoCapture('PA_7.avi')



i=0
k = False
poczatek = False
zapis = False

while(True):
    ret, ramka_org = wideo.read()
    
    if not ret:
        break
    
    ramka = ramka_org
        
    obraz1 = cv2.cvtColor(ramka,40)
        
    ramka_01 = cv2.inRange(obraz1, (0,0,100), (180,255,255))
    czern = cv2.inRange(obraz1, (0,0,0), (180,255,3))
        
    czern = czern[20:80,:]
    ramka_01 = ramka_01[20:80,:]
        
    x = len(ramka_01[:,0])
    y = len(ramka_01[0,:])
        
        
    cv2.imshow('oryginalny', ramka_org)
    cv2.imshow('obciety', ramka_org[20:80,:])
    
    sleep(0.05)
    
    d = cv2.waitKey(5) & 0xFF
    
    if d == 27 :
        break
    
    if np.sum(czern[59]) == 0:
        k= True
        
    if k == True and np.sum(ramka_01[0,:]) == 0 and np.sum(ramka_01[59,:]) == 0 and np.sum(ramka_01) != 0:
       cv2.imwrite(f'data1/klatka{i}.png',ramka_org[20:80,:]) 
       i=i+1
    cv2.imshow('oryginalny', ramka)
    
    
    
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break
 
wideo.release()       
cv2.destroyAllWindows()