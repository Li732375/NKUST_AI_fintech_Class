# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:29:13 2024

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_stock_data(stock_data, label_column, delete_column, time_step = 1, 
                     test_size = 0.3, random_state = 42):
    x, y = [], []
    scaled_data = stock_data.drop([label_column, delete_column], 
                                  axis = 1).values
    
    label = stock_data[label_column].values
    
    for i in range(time_step, len(scaled_data)):
        x.append(scaled_data[i - time_step : i])
        y.append(label[i])
    
    x = np.array(x)
    y = np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)
    return x_train, x_test, y_train, y_test