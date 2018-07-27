#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:44:46 2018

@author: ekele
"""

from knn_from_scratch import KNN
from util import get_xor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, Y = get_xor()
    
    plt.scatter(X[:,0],X[:,1], s = 100, c = Y, alpha = 0.5)
    plt.show()
    
    model = KNN(3)
    model.fit(X,Y)
    print('Accuracy:', model.score(X,Y))