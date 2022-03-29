# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:31:43 2022

@author: Filip
"""

import cv2
import numpy as np
from time import sleep
from projekt_v1 import  liczenie, zlapany,pokaz
import shutil
import os
from datetime import datetime as dt


czy_model = True
raport = f'Raport rozpoznawania obrazow \nProjekt Wizja2 \nData:{dt.now()}\n\n'
folder = 'Dane_wynik'
wideo = cv2.VideoCapture('PA_7.avi')
ref = cv2.imread('PA_7_ref.png')

if os.path.exists(folder):
    shutil.rmtree(folder) 
    
if os.path.exists(f'{folder}/raport.csv'):
    shutil.rmtree(f'{folder}/raport.csv')

os.mkdir(folder)

lista = np.zeros(18).astype('int')
lista2 = np.zeros(18).astype('int')
ob = liczenie(ref,lista)
cv2.imshow('wynik', ob)
if czy_model == True:
    cv2.imshow('wynik_model', ob)

i=0
k = False
poczatek = False
zapis = False

while(1):
    ret, ramka_org = wideo.read()
    if not ret:
        break
    ramka = cv2.GaussianBlur(ramka_org, (3,3), 2)
        
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
        
    if k == True and np.sum(ramka_01[0,:]) == 0 and np.sum(ramka_01[59,:]) == 0 and zapis == False and np.sum(ramka_01) != 0:
        lista, raport = zlapany(ramka_org,lista,lista2,i,raport,folder,czy_model)
        i=i+1
        zapis = True
        poczatek = False
        
    if np.isin(255,ramka_01[59,:]) == True and zapis == True:
        poczatek = True
        
    if np.isin(255,ramka_01[59,:]) == False and poczatek == True:
        zapis = False
            
            
        

      
cv2.destroyAllWindows()
wideo.release()
ref = cv2.imread('PA_7_ref.png')
new_ob = liczenie(ref, lista)
pokaz(new_ob)
f = open(f'{folder}/raport.csv', "w")
f.write(raport)
f.close()
