# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:06:54 2022

@author: Filip
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import colorsys
from test_sieci import model


from skimage.measure import label,regionprops




pd.options.display.float_format = "{:.2f}".format 


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
    
    c = cv2.cvtColor(o,40)
    b = cv2.inRange(c,(0,0,100),(255,255,255))
    
    cechy = regionprops(label(b))
    ile_obiektow = len(cechy)
    lista_cech = ['EulerNumber','Area','BoundingBoxArea','FilledArea','Extent','EquivDiameter','Solidity']
    ile_cech = len(lista_cech)
    tabela_cech = np.zeros((ile_obiektow,ile_cech+1+7+3)) 
    listaob = []
    for i in range(0,ile_obiektow):
        yp,xp,yk,xk = cechy[i]['BoundingBox']
        aktualny_obiekt = o[yp:yk,xp:xk,:]
        ret,binobj = cv2.threshold(aktualny_obiekt[:,:,1],0,255,cv2.THRESH_BINARY)      
        listaob.append(binobj) 
        
        for j in range(0,ile_cech):
            tabela_cech[i,j] = cechy[i][lista_cech[j]]
         
        hu = cv2.HuMoments(cv2.moments(binobj))
        hulog = (1 - 2*(hu>0).astype('int'))* np.nan_to_num(np.log10(np.abs(hu)),copy=True,neginf=-99,posinf=99)
        tabela_cech[i,ile_cech+1:ile_cech+8] = hulog.flatten()
        
        obiekt = pokazywanie_obiektow(o, [i for i in range(ile_obiektow)])
        
    for i in range(0,ile_obiektow):
        
        tabela_cech[i,15] = sredni_kolor(obiekt[i])[0]
        tabela_cech[i,16] = sredni_kolor(obiekt[i])[1]
        tabela_cech[i,17] = sredni_kolor(obiekt[i])[2]
    tabela_cech[:,ile_cech] = tabela_cech[:,3]/tabela_cech[:,2] 
    tabela_cech[:,0] = (tabela_cech[:,0] == 1) 
    return listaob, tabela_cech

def ekstrakcja_klas(o,klatki = True):
    c = cv2.cvtColor(o,40)
    b = cv2.inRange(c,(0,0,100),(255,255,255))
    
    cechy = regionprops(label(b))

    ile_obiektow = len(cechy)
    
    kolory = np.unique(o.reshape(-1, o.shape[2]), axis=0) # kolory w obrazie
    kolory = dzielenie_kolorow(kolory)
    
    kategorie1 = np.zeros((ile_obiektow,1)).astype('int')
    kategorie2 = np.zeros((ile_obiektow,1)).astype('int')
    kategorie3 = np.zeros((ile_obiektow,1)).astype('int')
    kategorie = np.zeros((ile_obiektow)).astype('int')
    lo,tc = ekstrakcja_cech(o)
    for i in range(ile_obiektow):
        if tc[i][4]>0.9:
            kategorie1[i]=1 
        else:
            kategorie1[i]=0 
            
        #print(f'ksztalt {tc[i,4]} -> {kategorie1[i]}')
        
    for k in range(len(kolory)):    
        kolory[k] = zamiana_bgr2hsv(kolory[k])
    
    for i in range(ile_obiektow):
        probka = np.array([tc[i][15],tc[i][16],tc[i][17]])
        
        #print(f"probka hsv - {probka}")
        probka = zamiana_bgr2hsv(probka)
        
        if(probka[0]<(0.33+0.05) and probka[0]>(0.33-0.05)):
            kategorie2[i] = 0   
        elif(probka[0]<(0.85+0.05) and probka[0]>(0.85-0.05)):
            kategorie2[i] = 1   
        elif(probka[0]<(0.0+0.05) and probka[0]>(0.0-0.05)):
            kategorie2[i] = 2   
        #print(f'kolor {probka[0]} -> {kategorie2[i]}')
        if klatki == True:
            if tc[i][1] > 500:
                kategorie3[i] = 0   
            elif tc[i][1] < 250:
                kategorie3[i] = 2   
            else:
                kategorie3[i] = 1   
        elif klatki == False:
            if tc[i][1] > 1100:
                kategorie3[i] = 0   
            elif tc[i][1] < 450:
                kategorie3[i] = 2   
            else:
                kategorie3[i] = 1   
        
        kategorie[i] = kategorie1[i]*9 + kategorie2[i]*3 + kategorie3[i]
        
                
    return kategorie


def dzielenie_kolorow(kolory):
    
    
    kolory = np.ndarray.tolist(kolory)
    k=0
    lista = []
    for i in range(len(kolory)):
        kolory[i]=colorsys.rgb_to_hsv(kolory[i][2], kolory[i][1], kolory[i][0])
    
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
    
    new = []
    for i in range(len(lista)):
        lol = colorsys.hsv_to_rgb(lista[i][0], lista[i][1], lista[i][2])
        lol2= [lol[2],lol[1],lol[0]]
    
        new.append(lol2)
    new = np.array(new)
   
    return new


def zamiana_bgr2hsv(kolory):
    kolory.astype('int')
    kolory = np.ndarray.tolist(kolory)
    
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
    c = cv2.cvtColor(o,40)
    b = cv2.inRange(c,(0,0,100),(255,255,255))
    cechy = regionprops(label(b))
    
    for i in range(len(lista)):
        yp,xp,yk,xk = cechy[lista[i]]['BoundingBox']
        aktualny_obiekt = o[yp:yk,xp:xk,:]
        wszystkie.append(aktualny_obiekt)
        
    return wszystkie
           


#sredni kolor
def sredni_kolor(obiekt):
    px = []
    ref = cv2.inRange(obiekt,(7,7,7),(255,255,255))
    x=len(ref)
    y=len(ref[0])
    #print(ref[42,41])
    #print(ref[:,0:10])
    #print(obiekt[:,0:10])
    for i in range(x-1):
        for j in range(y-1):
            if ref[i,j] >0:
                x = [obiekt[i,j]]
                px.append(x)
    #print(px[2])
    px = np.array(px).T
    kolor = []
    kolor.append(np.average(px[0]))
    kolor.append(np.average(px[1]))
    kolor.append(np.average(px[2]))
    kolor = np.array(kolor).astype('int')
    #print(kolor)
    return kolor
    

def zapisywanie(lo,tc,ka,tytul):
    np.savetxt(f"tc_{tytul}.csv",tc)
    np.savetxt(f"lo_{tytul}.csv",lo)
    np.savetxt(f"ka_{tytul}.csv",ka)
    
    


def dodawanie(lista,element):
    lista[element] = lista[element]+1
    return lista


def liczenie(ref,lista):
    ref2 = ref
    #szare duze kolo
    ref2 = cv2.putText(ref2,f'{lista[6]}',[15,90],1,3,[0,255,0])
    #szare srednie kolo
    ref2 = cv2.putText(ref2,f'{lista[7]}',[105,90],1,3,[0,255,0])
    #szare male kolo
    ref2 = cv2.putText(ref2,f'{lista[8]}',[190,90],1,3,[0,255,0])
    #szary duzy kwadrat
    ref2 = cv2.putText(ref2,f'{lista[15]}',[270,90],1,3,[0,255,0])
    #szary sredni kwadrat
    ref2 = cv2.putText(ref2,f'{lista[16]}',[370,90],1,3,[0,255,0])
    #szary maly kwadrat
    ref2 = cv2.putText(ref2,f'{lista[17]}',[470,90],1,3,[0,255,0])
    
    #zielone duze kolo
    ref2 = cv2.putText(ref2,f'{lista[0]}',[15,170],1,3,[0,0,255])
    #szare srednie kolo
    ref2 = cv2.putText(ref2,f'{lista[1]}',[105,170],1,3,[0,0,255])
    #szare male kolo
    ref2 = cv2.putText(ref2,f'{lista[2]}',[190,170],1,3,[0,0,255])
    #szary duzy kwadrat
    ref2 = cv2.putText(ref2,f'{lista[9]}',[270,170],1,3,[0,0,255])
    #szary sredni kwadrat
    ref2 = cv2.putText(ref2,f'{lista[10]}',[370,170],1,3,[0,0,255])
    #szary maly kwadrat
    ref2 = cv2.putText(ref2,f'{lista[11]}',[470,170],1,3,[0,0,255])
    
    #magenta duze kolo
    ref2 = cv2.putText(ref2,f'{lista[3]}',[15,270],1,3,[0,255,0])
    #szare srednie kolo
    ref2 = cv2.putText(ref2,f'{lista[4]}',[105,270],1,3,[0,255,0])
    #szare male kolo
    ref2 = cv2.putText(ref2,f'{lista[5]}',[190,270],1,3,[0,255,0])
    #szary duzy kwadrat
    ref2 = cv2.putText(ref2,f'{lista[12]}',[270,270],1,3,[0,255,0])
    #szary sredni kwadrat
    ref2 = cv2.putText(ref2,f'{lista[13]}',[370,270],1,3,[0,255,0])
    #szary maly kwadrat
    ref2 = cv2.putText(ref2,f'{lista[14]}',[470,270],1,3,[0,255,0])
    
    return ref2


def jaki_obiekt(idd):
    if idd == 0:
        text = 'kolo zielone duze'
    elif idd == 1: 
        text = 'kolo zielone srednie'
    elif idd == 2: 
        text = 'kolo zielone male'
    elif idd == 3: 
        text = 'kolo magenta duze'
    elif idd == 4:
        text = 'kolo magenta srednie'
    elif idd == 5:
        text = 'kolo magenta male'
    elif idd == 6: 
        text = 'kolo szare duze'
    elif idd == 7: 
        text = 'kolo szare srednie'
    elif idd == 8: 
        text = 'kolo szare male'
    elif idd == 9: 
        text = 'kwadrat zielony duzy'
    elif idd == 10:
        text = ' kwadrat zielony sredni'
    elif idd == 11:
        text = ' kwadrat zielony maly'
    elif idd == 12:
        text = ' kwadrat magenta duzy'
    elif idd == 13:
        text = ' kwadrat magenta sredni'
    elif idd == 14:
        text = ' kwadrat magenta maly'
    elif idd == 15:
        text = ' kwadrat szary duzy'
    elif idd == 16:
        text = ' kwadrat szary sredni'
    elif idd == 17:
        text = ' kwadrat szary maly'
    return text


def zlapany(ramka_org,lista,lista2,i,raport,folder = 'dane_klatki'):
    ref = cv2.imread('PA_7_ref.png')
    ref2 = cv2.imread('PA_7_ref.png')
    
    o = ramka_org[0:60,:]
    nazwa = f'{folder}/klatka{i}.png'
    
    
    
    new = ekstrakcja_klas(o)
    lista = dodawanie(lista,new[0])
    new_ob = liczenie(ref, lista)
    cv2.imshow('wynik', new_ob)
    
    _,dane = ekstrakcja_cech(o)
    new2 = model(dane)
    lista2 = dodawanie(lista2,new2[0])
    new_ob2 = liczenie(ref2, lista2)
    cv2.imshow('wynik_model', new_ob2)
    
    text1 = f'kształt z ekstrakcji: {jaki_obiekt(new[0])}'
    text3 = f'   kształt z modelu: {jaki_obiekt(new2[0])}'
    text2 = f'   Sciezka do klatki - {nazwa}'
    print(f'znaleziono {text1}')
    print(f'zapisywanie klatki. {text2}')
    raport = raport + f'{i}. ' + text1 + '\n' + text3 + '\n' + text2 + '\n'
    cv2.imwrite(nazwa, ramka_org[20:80,:])
    return lista, raport
        

    


    








        



    




