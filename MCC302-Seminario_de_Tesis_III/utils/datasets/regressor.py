#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import train_test_split

def getRegressSplit(X, y, r_s=None, validation=False):
  
  if validation:
    xtrain, xval, ytrain, yval = train_test_split(X, y, shuffle=True, test_size = 0.5, random_state=r_s)
    
    xval, xtest, yval, ytest = train_test_split(xval, yval, shuffle=True, test_size = 0.5, random_state=r_s)

    return xtrain, xval, xtest, ytrain, yval, ytest
    
  else:
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, shuffle=True, test_size = 0.25, random_state=r_s)    
  
    return xtrain, None, xtest, ytrain, None, ytest
