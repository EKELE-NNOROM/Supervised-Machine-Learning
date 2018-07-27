#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:37:21 2018

@author: ekele
"""

import numpy as np
import matplotlib.pyplot as plt

from knn_from_scratch import KNN

def get_data():
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1 
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return X, Y
    

if __name__ == '__main__':
            X, Y = get_data()
            
            plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha = 0.5)
            plt.show()
            
            model = KNN(3)
            model.fit(X,Y)
            print("Train accuracy:", model.score(X,Y))