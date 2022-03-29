# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:21:06 2022

@author: Filip
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow import keras
from projekt_v1 import ekstrakcja_cech,ekstrakcja_klas

lo = []
X = []
y = []
for i in range(0,1):
    try:
        o = cv2.imread(f"PA_76_ref.png")
        lo1, X1 = ekstrakcja_cech(o)
        y1 = np.ravel(ekstrakcja_klas(o))
        
    except:
        print('blad probka nr {i} nie jest prawidlowa!')

for i in range(18):
    lo.append(lo1[i])
    X.append(X1[i])
    y.append(y1[i])    


lo = np.array(lo)
X = np.array(X)
y = np.array(y)
yy = keras.utils.to_categorical(y,18)

model = keras.models.load_model('model')
print(len(y))
#for i in range(10):
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


