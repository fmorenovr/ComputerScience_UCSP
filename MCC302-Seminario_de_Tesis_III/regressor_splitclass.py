#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import json
import numpy as np
from numpy import genfromtxt

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from joblib import dump, load
from scipy.io import savemat, loadmat

from utils import verifyDir
from utils.datasets import evalClass, getRegressSplit
from utils.libsvm import getRegressMetrics, getRegressor
from utils.preprocessing import getFeatures

years = ["2011", "2013", "2019"]
metrics = ["safety", "wealthy", "uniquely"]
cities = ["Boston", "New York City"]
delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

C = np.logspace(7, -7, 15)

num_splits = 10

features_dir = "features/"

methods_type = "linear"
output_dir = "outputs/splitclass/regressor_"+methods_type+"/"

features = ['gist', "vgg16_places", 'vgg16_gap_places', 'vgg16', 'vgg16_gap']#, 'fisher']

methods = ['Lasso', 'Ridge', 'LinearRegression', 'LinearSVR', 'Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT", "HistGBDT"]#, 'SVR', 'MLP', 'SGD', 'NuSVR']

methods = ['Lasso', 'Ridge', 'LinearRegression', 'LinearSVR'] if methods_type == "linear" else ['Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT"]

stand = ['none']#, 'standard']
reduct = ['none']#, 'PCA']

auc_results = {'length': 0}
for method in methods:
  auc_results[method] = []

data_results={}
for feat in features:
  data_results[feat] = auc_results.copy()

for year in years:
  for city in cities:
    for metric in metrics:

      len_features = []

      for f in features:
      
        rof_dir = output_dir+str(year)+"/"+city+"/"+metric+"/"+f+"/"
        verifyDir(rof_dir)
      
        X_, Y_ = getFeatures(f, city=city, metric=metric, year=year)
        
        data_results[f]['length'] = X_.shape[1]
        
        for s in stand:
          if s == 'standard':
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_)
          else:
            X_s = X_.copy()

          for r in reduct:
            if r == 'PCA':
              pca = PCA(n_components=0.95, svd_solver='full')#PCA(n_components=n_components)
              X_r = pca.fit_transform(X_s)
              print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
              print("X.shape reducted: ", X.shape)
            else:
              X_r = X_s.copy()
          
            for delta_i in delta:
              X_m = X_r.copy()
              Y_m = Y_.copy()
            
              X, y = evalClass(X_m, Y_m, delta_i, type_m="regression")

              print("Delta:", delta_i, "X:", X.shape, "Y:", y.shape)
              
              xtrain_val, _, xtest, ytrain_val, _, ytest = getRegressSplit(X, y)
              
              x_ = np.concatenate([xtrain_val, xtest])
              y_ = np.concatenate([ytrain_val, ytest])
              
              kf = KFold(n_splits=num_splits)
        
              log_fname = open(rof_dir + "results_"+str(int(delta_i*100))+".csv", "w")
              log_fname.write("method,p_train,p_test,mse_train, mse_test,r2_train,r2_test\n")

              for m in methods:
                print("Method:", m)
              
                namefile_m =  s +"_" + r

                root_dir = rof_dir+m+"/"
                verifyDir(root_dir)
                
                #f_html.write('<tr><td><b>delta = {}</b></td>'.format(delta_i));
              
                name_dir = root_dir+str(int(delta_i*100))
                verifyDir(name_dir+"/")
                verifyDir(name_dir+"/logs/")
                verifyDir(name_dir+"/models/")
                verifyDir(name_dir+"/charts/")
                
                scores_p = []
                scores_mse = []
                scores_r2 = []
                split= 0
                for i_train, i_test in kf.split(x_):
                  
                  xtrain_val, xtest, ytrain_val, ytest = x_[i_train], x_[i_test], y_[i_train], y_[i_test]
                
                  svr = getRegressor(m, 1e-15)

                  svr_model = svr.fit(xtrain_val, ytrain_val)
                  
                  p_train, mse_train, r2_train = getRegressMetrics(svr_model, xtrain_val, ytrain_val)
                  p_test, mse_test, r2_test = getRegressMetrics(svr_model, xtest, ytest)
                    
                  print("Split ",split,": Train:", p_train, "Test:", p_test)
                
                  scores_p.append(p_test)
                  scores_mse.append(mse_test)
                  scores_r2.append(r2_test)
                  split=split+1
                
                scores_p = np.asarray(scores_p)
                scores_mse = np.asarray(scores_mse)
                scores_r2 = np.asarray(scores_r2)
                
                print("Resume test: Pearson:", scores_p.mean(), "MSE:", scores_mse.mean(), "R2:",scores_r2.mean())
                
                log_fname.write(m+","+ str(round(p_train,5)) +"," +str(round(scores_p.mean(),5))+","+str(round(mse_train,5))+","+ str(round(scores_mse.mean(),5))+","+str(round(r2_train,5))+","+str(round(scores_r2.mean(),5))+"\n")
                
              log_fname.close()
                
