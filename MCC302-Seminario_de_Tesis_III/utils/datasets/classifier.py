#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import train_test_split

def getClassSplit(X, y, random_state=None):
  y_pos = y[y==1]
  y_neg = y[y!=1]
  
  index_pos = np.where(y==1)[0]
  index_neg = np.where(y!=1)[0]
  
  X_pos = X[index_pos]
  X_neg = X[index_neg]
  
  xtrain_pos, xval_pos, ytrain_pos, yval_pos = train_test_split(X_pos, y_pos, shuffle=True, test_size = 0.5, random_state=random_state)
  xtrain_neg, xval_neg, ytrain_neg, yval_neg = train_test_split(X_neg, y_neg, shuffle=True, test_size = 0.5, random_state=random_state)
                  
  xval_pos, xtest_pos, yval_pos, ytest_pos = train_test_split(xval_pos, yval_pos, shuffle=True, test_size = 0.5, random_state=random_state)
  xval_neg, xtest_neg, yval_neg, ytest_neg = train_test_split(xval_neg, yval_neg, shuffle=True, test_size = 0.5, random_state=random_state)
                  
  xtrain = np.concatenate([xtrain_pos, xtrain_neg])
  xval = np.concatenate([xval_pos, xval_neg])
  xtest = np.concatenate([xtest_pos, xtest_neg])
  
  ytrain = np.concatenate([ytrain_pos, ytrain_neg])
  yval = np.concatenate([yval_pos, yval_neg])
  ytest = np.concatenate([ytest_pos, ytest_neg])

  return xtrain, xval, xtest, ytrain, yval, ytest
