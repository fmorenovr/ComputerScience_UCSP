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
output_dir = "outputs/kfold/regressor_"+methods_type+"/"

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
              
              kf = KFold(n_splits=num_splits)

              log_fname = open(rof_dir + "results_"+str(int(delta_i*100))+".csv", "w")
              log_fname.write("method,c,p_train,p_test,mse_train, mse_test,r2_train,r2_test\n")

              for m in methods:
                print("Method:", m)
                print("X:", xtrain_val.shape, "Y:", ytrain_val.shape)
              
                namefile_m =  s +"_" + r

                root_dir = rof_dir+m+"/"
                verifyDir(root_dir)
                
                #f_html.write('<tr><td><b>delta = {}</b></td>'.format(delta_i));
              
                name_dir = root_dir+str(int(delta_i*100))
                verifyDir(name_dir+"/")
                verifyDir(name_dir+"/logs/")
                verifyDir(name_dir+"/models/")
                verifyDir(name_dir+"/charts/")
                
                #for split in range(num_splits):
                scores_global_val =[]
                scores_global_train = []
                split= 0
                for i_train, i_val in kf.split(xtrain_val):
                  
                  xtrain, xval, ytrain, yval = xtrain_val[i_train], xtrain_val[i_val], ytrain_val[i_train], ytrain_val[i_val]
                  
                  #print("Split:", split)
                  print("xtrain:", xtrain.shape, "xval:", xval.shape, "xtest:", xtest.shape)
                  print("ytrain:", ytrain.shape, "yval:", yval.shape, "ytest:", ytest.shape)
                  
                  fname_split = open(name_dir + "/logs/"+namefile_m+"_"+str(split)+".csv", "w")
                  fname_split.write("c,train,val\n")
                  
                  scores_val = []
                  scores_train = []
                  
                  for c in C:
                    print("Evaluating", f, "with method", m, "with c=",c)
                    svr = getRegressor(m, c)
                    svr_model = svr.fit(xtrain, ytrain)
                    
                    # R, mse, mrsq
                    p_train, _, _ = getRegressMetrics(svr_model, xtrain, ytrain)
                    p_val, _, _ = getRegressMetrics(svr_model, xval, yval)

                    scores_val.append(p_val)
                    scores_train.append(p_train)
                    
                    print("train:", p_train, "val:", p_val)
                    
                    fname_split.write(str(c) + ","+str(round(p_train,5)) +"," +str(round(p_val,5)) +"\n")
                    
                  fname_split.close()
                  scores_global_val.append(scores_val)
                  scores_global_train.append(scores_train)
                  split=split+1
                
                scores_global_val = np.asarray(scores_global_val)
                scores_means = np.nanmean(scores_global_val, axis=0)
                
                scores_global_train = np.asarray(scores_global_train)
                scores_means_train = np.nanmean(scores_global_train, axis=0)
                
                fname_max = open(name_dir + "/means.csv", "w")
                fname_max.write("split,c,train,test\n")
                for indx in range(len(scores_means)):
                  fname_max.write(str(C[indx])+","+ str(round(scores_means_train[indx],5)) +"," +str(round(scores_means[indx],5)) +"\n")
                  
                fname_max.close()
                
                nan_array = np.isnan(scores_means)
                not_nan_array = ~ nan_array
                scores_means = scores_means[not_nan_array]
                C_ = C[not_nan_array]
                
                index_max = np.argmax(scores_means)
                c_val = C_[index_max]
                
                svr = getRegressor(m, c_val)

                svr_model = svr.fit(xtrain_val, ytrain_val)

                p_train, mse_train, r2_train = getRegressMetrics(svr_model, xtrain_val, ytrain_val)
                
                p_test, mse_test, r2_test = getRegressMetrics(svr_model, xtest, ytest)
                  
                ypred = svr_model.predict(xtest)
                  
                plt.figure()
                plt.plot(ytest, ytest,'ro')
                plt.plot(ytest, ypred,'bo', alpha=0.1)
                plt.yticks(np.arange(0, 10, step=0.5))
                plt.xticks(np.arange(0, 10, step=0.5))
                plt.title("Regresion pearson="+str(round(p_test,5))+", c="+str(c_val))
                plt.legend(["valor real", "valor predicho"])
                plt.grid(True)
                plt.savefig(name_dir + "/charts/"+namefile_m+".png")
                plt.clf()
                plt.cla()
                plt.close()
                print("Resume: Train:", p_train, "Test:", p_test)
                
                log_fname.write(m+","+str(c_val)+","+ str(round(p_train,5)) +"," +str(round(p_test,5))+","+str(round(mse_train,5))+","+ str(round(mse_test,5))+","+str(round(r2_train,5))+","+str(round(r2_test,5))+"\n")
                dump(svr_model, name_dir+"/models/" + m+'.joblib')
                
              log_fname.close()
                
