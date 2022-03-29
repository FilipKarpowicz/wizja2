# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:39:19 2022

@author: Filip
"""

import numpy as np
from tensorflow import keras
import cv2
from projekt_v1 import ekstrakcja_cech, model,ekstrakcja_klas




o = cv2.imread(f"PA_76_ref.png")
lo,tc = ekstrakcja_cech(o)
ka = ekstrakcja_klas(o)
wynik = model(tc)

print(wynik)
print(ka)
