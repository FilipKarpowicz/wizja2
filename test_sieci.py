# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:39:19 2022

@author: Filip
"""

import numpy as np
from tensorflow import keras

  
def model(dane):
    model = keras.models.load_model('model')
    y_pred = model.predict(dane)
    y_pred_max = np.amax(y_pred,1)
    y_pred_id = np.array([np.argwhere(y_pred==maxval).flatten()[1] for maxval in y_pred_max])
    return y_pred_id

