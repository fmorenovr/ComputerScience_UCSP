#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GridSearchCV

from scipy.stats import pearsonr

from utils import verifyFile, verifyDir

class MidpointNormalize(Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    Normalize.__init__(self, vmin, vmax, clip)

  def __call__(self, value, clip=None):
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y))

metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]

kernel_name="rbf"

name_dir = "SVR_RBF/"

PCA_method = True
standard_method = True

if PCA_method:
  name_dir= name_dir+"PCA/"

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

    C_range = np.logspace(-7, 7, 15)
    gamma_range = np.logspace(-9, 3, 13)
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25)
    
    scores_test = []
    scores_train = []
    for g in gamma_range:
      pearson_train = []
      pearson_test = []
      
      for c in C_range:
        print("Using kernel {}, c={} and gamma={}".format("rbf", c, g))
        svr = SVR(kernel=kernel_name, C=c, gamma=g)
        svr_model = svr.fit(xtrain, ytrain)
        
        ypred = svr_model.predict(xtest)
        pearson_coef, _ = pearsonr(ytest, ypred)
        print("Pearson correlation test is: ", pearson_coef)
        print("R2 score is: ", svr_model.score(xtest, ytest), "\n")
        
        ppar = math.isnan(pearson_coef)
        if ppar:
          pearson_coef = 0
        
        plt.figure()
        plt.plot(ytest, ytest,'r')
        plt.plot(ytest, ypred,'bo', alpha=0.1)
        plt.yticks(np.arange(0, 10, step=0.5))
        plt.xticks(np.arange(0, 10, step=0.5))
        plt.title("Regresion y-test pearson="+str(round(pearson_coef,5)))
        plt.legend(["valor real", "valor predicho"])
        plt.grid(True)
        plt.savefig(name_dir+"ytest_"+str(c)+"_"+str(g)+"_"+city+"_"+metric+".png")
        plt.clf()
        plt.cla()
        plt.close()
        
        ytrain_pred = svr_model.predict(xtrain)      
        pearson_coef_train, _ = pearsonr(ytrain, ytrain_pred)
        print("Pearson correlation train is: ", pearson_coef_train)
        print("R2 score is: ", svr_model.score(xtrain, ytrain), "\n")
        
        ppar = math.isnan(pearson_coef_train)
        if ppar:
          pearson_coef_train = 0
        
        plt.figure()
        plt.plot(ytrain, ytrain,'r')
        plt.plot(ytrain, ytrain_pred,'bo', alpha=0.1)
        plt.yticks(np.arange(0, 10, step=0.5))
        plt.xticks(np.arange(0, 10, step=0.5))
        plt.title("Regresion ytrain pearson="+str(round(pearson_coef_train,5)))
        plt.legend(["valor real", "valor predicho"])
        plt.grid(True)
        plt.savefig(name_dir+"ytrain_"+str(c)+"_"+str(g)+"_"+city+"_"+metric+".png")
        plt.clf()
        plt.cla()
        plt.close()
        
        pearson_train.append(pearson_coef_train)
        pearson_test.append(pearson_coef)
      
      plt.figure()
      plt.plot(C_range, pearson_train,'r')
      plt.plot(C_range, pearson_test,'b')
      plt.yticks(np.arange(0, 1, step=0.05))
      plt.xticks(C_range)
      plt.xscale('log')
      plt.xlabel('C')
      plt.ylabel('Pearson coeff')
      plt.title("Pearson coefficient vs C")
      plt.legend(["train", "test"])
      plt.grid(True)
      
      plt.savefig(name_dir+str(g)+"_"+city+"_"+metric+".png")
      
      plt.clf()
      plt.cla()
      plt.close()
        
      scores_train.append(pearson_train)
      scores_test.append(pearson_test)
    
    scores_train = np.asarray(scores_train)
    scores_test = np.asarray(scores_test)
    
    scores_train = scores_train.transpose()
    scores_test = scores_test.transpose()
    
    scores_train[scores_train < 0] = 0
    scores_test[scores_test < 0] = 0

    score_max = np.amax(scores_test)
    c_max, g_max = np.where(scores_test == score_max)
    print("The best parameters test are c={} and gamma={} with a score of {}" .format (C_range[c_max], gamma_range[g_max], score_max))
    
    score_max = np.amax(scores_train)
    c_max, g_max = np.where(scores_train == score_max)
    print("The best parameters train are c={} and gamma={} with a score of {}" .format (C_range[c_max], gamma_range[g_max], score_max))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_test, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy test')#; PCA components='+str(n_features))
    plt.savefig(name_dir+"cmap_test.png")
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_train, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy train')#; PCA components='+str(n_features))
    plt.savefig(name_dir+"cmap_train.png")
    plt.clf()
    plt.cla()
    plt.close()
