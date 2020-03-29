#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score

from scipy.stats import pearsonr

from utils import verifyFile, verifyDir

C = np.logspace(-7, 7, 15)
metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]
cities_test = ["Boston", "New York City"]

name_dir = "LinearSVR/"

PCA_method = True
SVR_method = True
standard_method = True

if PCA_method and SVR_method:
  name_dir= name_dir+"PCA_SVR/"
elif PCA_method:
  name_dir= name_dir+"PCA/"
elif SVR_method:
  name_dir= name_dir+"SVR/"

if standard_method:
  name_dir= name_dir+"standard/"

verifyDir(name_dir)

for city in cities:
  for metric in metrics:
    print("preparing {}-{} data ...".format(city, metric))
    X = genfromtxt("X_"+city+".csv", delimiter=',')
    y = genfromtxt("y_"+city+".csv", delimiter=',')
    print(X.shape)
    print(y.shape)

    if PCA_method:
      pca = PCA(n_components=0.95, svd_solver='full')
      X = pca.fit_transform(X)
      n_components, n_features = X.shape
      print(X.shape)
      print(y.shape)
   
    if standard_method:
      scaler = StandardScaler()
      X = scaler.fit_transform(X)

    print("Preparing train and test ...")
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25)
    
    pearsons_test = []
    pearsons_train = []
    
    print("Fitting SVR+Linear ...")
    for c in C:
      print("Using kernel {} and c={}".format("linear", c))
      if SVR_method:
        svr = SVR(kernel='linear', C=c)
      else:
        svr = LinearSVR(C=c)
      
      svr_model = svr.fit(xtrain, ytrain)
      
      ypred = svr_model.predict(xtest)
      pearson_coef, _ = pearsonr(ytest, ypred)
      print("Pearson correlation test is: ", pearson_coef)
      print("R2 score is: ", svr_model.score(xtest, ytest), "\n")
      
      plt.figure()
      plt.plot(ytest, ytest,'r')
      plt.plot(ytest, ypred,'bo', alpha=0.1)
      plt.yticks(np.arange(0, 10, step=0.5))
      plt.xticks(np.arange(0, 10, step=0.5))
      plt.title("Regresion y-test pearson="+str(round(pearson_coef,5)))
      plt.legend(["valor real", "valor predicho"])
      plt.grid(True)
      plt.savefig(name_dir+"ytest_"+str(c)+"_"+city+"_"+metric+".png")
      plt.clf()
      plt.cla()
      plt.close()
      
      ytrain_pred = svr_model.predict(xtrain)      
      pearson_coef_train, _ = pearsonr(ytrain, ytrain_pred)
      print("Pearson correlation train is: ", pearson_coef_train)
      print("R2 score is: ", svr_model.score(xtrain, ytrain), "\n")
      
      plt.figure()
      plt.plot(ytrain, ytrain,'r')
      plt.plot(ytrain, ytrain_pred,'bo', alpha=0.1)
      plt.yticks(np.arange(0, 10, step=0.5))
      plt.xticks(np.arange(0, 10, step=0.5))
      plt.title("Regresion ytrain pearson="+str(round(pearson_coef_train,5)))
      plt.legend(["valor real", "valor predicho"])
      plt.grid(True)
      plt.savefig(name_dir+"ytrain_"+str(c)+"_"+city+"_"+metric+".png")
      plt.clf()
      plt.cla()
      plt.close()
      
      pearsons_test.append(pearson_coef)
      pearsons_train.append(pearson_coef_train)
    
    plt.figure()
    plt.plot(C, pearsons_train,'r')
    plt.plot(C, pearsons_test,'b')
    plt.yticks(np.arange(0, 1, step=0.05))
    plt.xticks(C)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Pearson coeff')
    plt.title("Pearson coefficient vs C")
    plt.legend(["train", "test"])
    plt.grid(True)
    
    plt.savefig(name_dir+city+"_"+metric+".png")
    
    plt.clf()
    plt.cla()
    plt.close()
    
    '''
    print("Fitting cross-validation")
    
    shuffle_fold = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    
    for c in C:
      svr = SVR(kernel='linear', C=c)
      print("Using kernel {} and c={}".format("linear", c)) 
      
      scores = [svr.fit(X[train], y[train]).score(X[test], y[test]) for train, test in shuffle_fold.split(X)]
    
      print("Scores: ", scores)
      scores = np.asarray(scores)
      print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2), "\n")
    '''
