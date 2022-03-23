# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:06:54 2022

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

# zmiana sposobu wyświetlania danych typu float
pd.options.display.float_format = "{:.2f}".format 

#FUNKCJE POMOCNICZE

# funkcja wyswietlająca obraz kolorowy lub w skali szarości
def pokaz(obraz, tytul = "", osie = True, openCV = True, colmap = 'gray'):
    if not(osie):
        plt.axis("off") 
    if obraz.ndim == 2:
        plt.imshow(obraz,cmap = colmap,vmin=0, vmax=255)
    else:
        if openCV:
            plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2RGB),vmin=0, vmax=255)
        else:
            plt.imshow(obraz,interpolation = 'none',vmin=0, vmax=255)
    plt.title(tytul)

def polob(listaobr, ile_k = 1, listatyt = [], openCV = True, wart_dpi = 100, osie = False, colmap = 'gray'):
    rozm_obr = 5
    ile = len(listaobr)
    if len(listatyt) == 0:
        listatyt = [' ']*ile
    ile_w = np.ceil(ile / ile_k).astype(int)
    figsize_k = rozm_obr*ile_k
    figsize_w = rozm_obr*ile_w
    plt.figure(figsize=(figsize_k,figsize_w), dpi = wart_dpi)
    for i in range(0,ile):
        if isinstance(listaobr[i],np.ndarray):
            plt.subplot(ile_w,ile_k,i+1)
            pokaz(listaobr[i],listatyt[i],osie,openCV, colmap)
    plt.show()




def ekstrakcja_cech(o):
    # ekstrakcja cech
    # binaryzacja obrazu
    b = cv2.inRange(o,(1,1,1),(255,255,255))
    # etykietowanie i ekstrakcja cech
    cechy = regionprops(label(b))
    ile_obiektow = len(cechy)
    lista_cech = ['EulerNumber','Area','BoundingBoxArea','FilledArea','Extent','EquivDiameter','Solidity']
    ile_cech = len(lista_cech)
    tabela_cech = np.zeros((ile_obiektow,ile_cech+1+7)) # "1" - to jedna cecha wyliczna, "7" to momenty Hu
    listaob = []
    for i in range(0,ile_obiektow):
        yp,xp,yk,xk = cechy[i]['BoundingBox']
        aktualny_obiekt = o[yp:yk,xp:xk,:]
        ret,binobj = cv2.threshold(aktualny_obiekt[:,:,1],0,255,cv2.THRESH_BINARY)      
        listaob.append(binobj) #aktualny_obiekt)
        # rejestrujemy wybrane cechy wyznaczone przez regionprops
        for j in range(0,ile_cech):
            tabela_cech[i,j] = cechy[i][lista_cech[j]]
        # dodajemy momenty Hu   
        hu = cv2.HuMoments(cv2.moments(binobj))
        hulog = (1 - 2*(hu>0).astype('int'))* np.nan_to_num(np.log10(np.abs(hu)),copy=True,neginf=-99,posinf=99)
        tabela_cech[i,ile_cech+1:ile_cech+8] = hulog.flatten()
    tabela_cech[:,ile_cech] = tabela_cech[:,3]/tabela_cech[:,2] # cecha wyliczana
    tabela_cech[:,0] = (tabela_cech[:,0] == 1) # korekta liczby Eulera
    return listaob, tabela_cech

def ekstrakcja_klas(o):
    # ekstrakcja kategorii
    # binaryzacja obrazu
    b = cv2.inRange(o,(1,1,1),(255,255,255))
    # etykietowanie i ekstrakcja cech
    cechy = regionprops(label(b))

    ile_obiektow = len(cechy)
    # wyszukiwanie kolorów
    kolory = np.unique(o.reshape(-1, o.shape[2]), axis=0) # kolory w obrazie
    kolory = dzielenie_kolorow(kolory)
    #print(kolory)
    #wyszukiwanie kształtow 
    
    kategorie1 = np.zeros((ile_obiektow,1)).astype('int')
    kategorie2 = np.zeros((ile_obiektow,1)).astype('int')
    
    lo,tc = ekstrakcja_cech(o)
    for i in range(ile_obiektow):
        if tc[i][4]>0.9:
            kategorie1[i]=1 #kwadraty
        else:
            kategorie1[i]=0 #koła
    #print(kolory)
    for k in range(len(kolory)):    
        kolory[k] = zamiana_bgr2hsv(kolory[k])
    #print(kolory) 
    for i in range(ile_obiektow):
        # wsp. jednego z punktów obiektu - do próbkowania koloru
        x,y = np.where(lo[i] == 255)
        #print(f"dlugosc lo = {len(lo)}")
        wartosc = x[0]*len(lo[i][0])+y[0]
        x,y = cechy[i]['Coordinates'][wartosc]
        probka = o[x,y]
        #print(i)
        probka = zamiana_bgr2hsv(probka)
        #print(f"probka hsv - {probka}")
        for j in range(len(kolory)):
            #print(len(kolory))
            if(probka[0]<(kolory[j][0]+0.05) and probka[0]>(kolory[j][0]-0.05)):
                kategorie2[i] = kategorie1[i]*len(kolory)+j
                #print(f" ksztalt nr {i+1} kategoria1 {kategorie1[i]} kolor {kolory[j]} nr koloru {j} kategoria2 {kategorie2[i]}")
              
              
            
        
    return kategorie2


def dzielenie_kolorow(kolory):
    #kolory hsv: h -> (0,1) | s-> (0,1) |v-> (0,255)
    # kolory rgb/bgr (0,255) (0,255) (0,255)
    #rbg\bgr [0,0,0] <-czarne , [255,255,255] <- biale
    
    kolory = np.ndarray.tolist(kolory)
    k=0
    lista = []
    for i in range(len(kolory)):
        kolory[i]=colorsys.rgb_to_hsv(kolory[i][2], kolory[i][1], kolory[i][0])
    #print(kolory[1:5])
    for i in range(len(kolory)):
        k=0
        if(kolory[i][2]>50):
            if len(lista) == 0:
                lista.append(kolory[i])
            else:
                for j in range(len(lista)):
                    
                    if(kolory[i][0]<(lista[j][0]+0.01) and kolory[i][0]>(lista[j][0]-0.010)):
                        k=1           
                if k==0:
                    lista.append(kolory[i])
    #print(lista)
    new = []
    for i in range(len(lista)):
        lol = colorsys.hsv_to_rgb(lista[i][0], lista[i][1], lista[i][2])
        lol2= [lol[2],lol[1],lol[0]]
        #lol = np.array(lol)
        #print(f"loool ->{lol}")
        new.append(lol2)
    new = np.array(new)
    #uwaga kolory wychodza w rgb  a powinny w bgr
    return new


def zamiana_bgr2hsv(kolory):
    kolory.astype('int')
    kolory = np.ndarray.tolist(kolory)
    #print(f"kolory81 {kolory}")
    kolory = colorsys.rgb_to_hsv(kolory[2], kolory[1], kolory[0])
    new = np.array(kolory)
    return new

def zamiana_hsv2bgr(kolory):
    kolory = np.ndarray.tolist(kolory)
    new = []
    for i in range(len(kolory)):
        lol = colorsys.hsv_to_rgb(kolory[i][0], kolory[i][1], kolory[i][2])
        lol2= [lol[2],lol[1],lol[0]]
        new.append(lol2)
    new = np.array(new)
    return(new)


def pokazywanie_obiektow(o,lista):
    wszystkie =[]
    b = cv2.inRange(o,(1,1,1),(255,255,255))
    # etykietowanie i ekstrakcja cech
    cechy = regionprops(label(b))
    lo,tc = ekstrakcja_cech(o)
    for i in range(len(lista)):
        yp,xp,yk,xk = cechy[lista[i]]['BoundingBox']
        aktualny_obiekt = o[yp:yk,xp:xk,:]
        wszystkie.append(aktualny_obiekt)
    polob(wszystkie,colmap='RGB',ile_k=3,osie=(True))
    return wszystkie
           


#sredni kolor
def sredni_kolor(obiekt,cechy):
    x,y = cechy[i]['Coordinates']
    


def skrypt():
    # wczytanie obrazu
    o = cv2.imread("PA_72_ref.png")
    b = cv2.inRange(o,(1,1,1),(255,255,255))
    # etykietowanie
    etykiety = label(b)
    #polob([o,b,etykiety],3)
    # wyznaczanie cech
    cechy = regionprops(etykiety)
    
    lo,tc = ekstrakcja_cech(o)
    ka = ekstrakcja_klas(o)
    pokaz(o,colmap='rgb')
    for j in range(6):
        lista_klasy = [lo[i] for i in np.where(ka == j)[0]]
        print("klasa:",j," obiektów:", len(lista_klasy))
        #polob(lista_klasy,11,colmap='winter')    
        x = np.where(ka == j)
        x = np.ndarray.tolist(x[0])
        pokazywanie_obiektow(o,x)





        



    




