# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:21:28 2022

@author: Filip
"""

import numpy as np
import cv2
import os
import shutil


wideo = cv2.VideoCapture('PA_7.avi')

i=0

if os.path.exists('data'):
    shutil.rmtree('data') 

os.mkdir('data')

k=0
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
    
    d = cv2.waitKey(1) & 0xFF
    if d == 27 :
        break
    
    if np.sum(czern[59]) == 0:
        k= True
        
    if k == True and np.sum(ramka_01[0,:]) == 0 and np.sum(ramka_01[59,:]) == 0 and np.sum(ramka_01) != 0:
       cv2.imwrite(f'data/klatka{i}.png',ramka_org[20:80,:]) 
       i=i+1
       
    
 
wideo.release()       
cv2.destroyAllWindows()