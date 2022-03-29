# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:36:32 2022

@author: Filip
"""

from keras.models import Sequential
from keras.layers import Dense


model = Sequential()

model = Sequential()
model.add(Dense(180, input_dim=18, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(18, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.save('model')





