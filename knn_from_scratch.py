#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:52:03 2018

@author: ekele
"""

import numpy as np
#from future.utils import iteritems
from sortedcontainers import SortedList

from util import get_data
from datetime import datetime
import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y): # X and y here are Xtrain and ytrain
        self.X = X
        self.y = y
        
    def predict(self,X): # X here is the Xtest
        y = np.zeros(len(X))
        for i, x in enumerate(X): # Iterate through test points
            sl = SortedList()
            for j, xt in enumerate(self.X): # Iterate through training points
                diff = x - xt
                d = diff.dot(diff) # square distance, square distance is monotonically increasing
                if len(sl) < self.k: # if length of sorted list is less than size k
                    sl.add((d, self.y[j]))#add current point without checking anything
                else:
                    if d < sl[-1][0]: # if current distance is less than the current value, delete the distance
                        del sl[-1]
                        sl.add((d, self.y[j]))
            
            votes = {}
            for _,v in sl:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0
            max_votes_class = -1
            for v,count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y
    
    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(P == Y)
            
if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1300
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    test_scores = []
    train_scores = []
    ks = (1,2,3,4,5)
    for k in ks:
        print('\nk = ', k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))
        
        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        print("Training accuracy: ", train_score)
        print("Time to compute train accuracy: ", (datetime.now() - t0), "Train size:", len(Ytrain))
        train_scores.append(train_score)
        
        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy: ", test_score)
        print("Time to compute test accuracy: ", (datetime.now() - t0), "Test size:", len(Ytest))
        test_scores.append(test_score)
        
    plt.plot(ks, train_scores, label='Train Scores')
    plt.plot(ks, test_scores, label='Test Scores')
    plt.legend()
    plt.show()
        