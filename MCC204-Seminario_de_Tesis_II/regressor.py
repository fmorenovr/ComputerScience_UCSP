#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from scipy.io import savemat, loadmat

import numpy as np
from numpy import genfromtxt
import pandas as pd
import random

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.metrics.scorer import make_scorer, roc_auc_score

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from utils import verifyFile, verifyDir

years = ["2011"]

name_dir = "regressor/"
verifyDir(name_dir)

C = np.logspace(7, -7, 15)
threshold = 0.015
n_components = 2

metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]

features = ['fisher']
methods = ['SGD', 'ridge', 'lasso', 'linearRegression', 'linearSVR']#, 'SVR', 'NuSVR']
stand = ['none', 'standard']
reduct = ['none', 'PCA']

def Pearson(y_true, y_pred, **kwargs):
  pearson_coef, _ = pearsonr(y_true,y_pred)
  return pearson_coef

def ROC_AUC(y_true, y_pred, **kwargs):
  ruc = roc_auc_score(y_true,y_pred)
  return ruc

my_scorer = make_scorer(Pearson, greater_is_better=True)

def getNumbers(y):
  y = y/10.0
  return y

for data_year in years:
  mat = loadmat('features_'+data_year+'.mat')
  gist = mat["gist_feature_matrix"]
  fisher = mat["fisher_feature_matrix"]
  image_names = mat["image_list"]
  scores = mat["scores"][0]

  y_scores = np.asarray(scores)
  X_gist = np.asarray(gist)
  X_fisher = np.asarray(fisher)
  print("y.shape: ", y_scores.shape)
  print("X_gist.shape: ", X_gist.shape)
  print("X_fisher.shape: ", X_fisher.shape)

  X_vgg16 = genfromtxt("X_"+"Boston"+"_"+data_year+".csv", delimiter=',')
  y_vgg16 = genfromtxt("y_"+"Boston"+"_"+data_year+".csv", delimiter=',')
  print("X_vgg16.shape: ", X_vgg16.shape)
  print("y_vgg16.shape: ", y_vgg16.shape)

  namefile_orig = data_year+"_"

  for f in features:
    namefile_f = namefile_orig
    if f == 'vgg16':
      X = X_vgg16
      y = y_vgg16
      print("Selected VGG16")
    else:
      if f == 'gist':
        X = X_gist
        print("Selected GIST")
      else:
        X = X_fisher
        print("Selected FISHER")
      y = y_scores
    y = getNumbers(y)
    for s in stand:
      namefile_s = namefile_f + s + "_"
      if s == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

      for r in reduct:
        namefile_r = namefile_s + r
        if r == 'PCA':
          pca = PCA(n_components=0.95, svd_solver='full')#PCA(n_components=n_components)
          X = pca.fit_transform(X)
          print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
          print("X.shape reducted: ", X.shape)

        #shuffle_fold = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
        #xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.6)

        k_fold = KFold(n_splits=10, shuffle=True)
        for m in methods:
          namefile_m = m + "_" + namefile_r
          fname = open(name_dir + "/"+f+"_"+namefile_m+".csv", "w")
          fname.write("c,train,test\n")
          
          for c in C:
            print("Evaluating ",m, " with c=",c)
            if m == "SGD":
              svr = SGDRegressor(alpha=c)
            elif m == "lasso":
              svr = Lasso(alpha=c)
            elif m == "ridge":
              svr = Ridge(alpha=c)
            elif m == "linearRegression":
              svr = LinearRegression()
            elif m == "linearSVR":
              svr = LinearSVR(C=c)
            elif m == "SVR":
              svr = SVR(kernel='linear', C=c)
            elif m == "NuSVR":
              svr = NuSVR(kernel='linear', C=c, nu=0.1)
            else:
              svr = LinearSVR(C=c)
            
            #scores = cross_val_score(svr, X, y, scoring=my_scorer, cv=10, n_jobs=-1)

            scores_train = []
            scores_test = []
            for train_index, test_index in k_fold.split(X):
              xtrain, xtest, ytrain, ytest = X[train_index], X[test_index], y[train_index], y[test_index]
              
              svr_model = svr.fit(xtrain, ytrain)

            #  z = svr_model.coef_[index_0]*xx + svr_model.coef_[index_1]*yy + svr_model.intercept_
              
              ypred = svr_model.predict(xtest)
              metric_test, _ = pearsonr(ytest, ypred)
              #print("Pearson correlation test is: ", metric_test)
              #print("R2 score is: ", svr_model.score(xtest, ytest), "\n")

              ytrain_pred = svr_model.predict(xtrain)      
              metric_train, _ = pearsonr(ytrain, ytrain_pred)
              #print("Pearson correlation train is: ", metric_train)
              #print("R2 score is: ", svr_model.score(xtrain, ytrain), "\n")
            
              scores_train.append(metric_train)
              scores_test.append(metric_test)
            
            scores_train = np.asarray(scores_train)
            scores_test = np.asarray(scores_test)
            
            print("Mean:", scores_train, scores_test)
            print("Mean:", scores_train.mean(), scores_test.mean())
            fname.write(str(c) + ","+ str(round(scores_train.mean(),5)) +"," +str(round(scores_test.mean(),5)) +"\n")
          fname.close()
          
