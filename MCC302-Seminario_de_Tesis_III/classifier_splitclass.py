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

from utils import verifyDir
from utils.datasets import evalClass, getClassSplit
from utils.libsvm import getClassifier, getClassMetrics
from utils.preprocessing import getFeatures

years = ["2011", "2013", "2019"]
metrics = ["safety", "wealthy", "uniquely"]
cities = ["Boston", "New York City"]
delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

C = np.logspace(7, -7, 15)

num_splits = 10

features_dir = "features/"

methods_type = "linear"
output_dir = "outputs/splitclass/classifier_"+methods_type+"/"

features = ['gist', "vgg16_places", 'vgg16_gap_places', 'vgg16', 'vgg16_gap']#, 'fisher']

methods = ['RidgeClassifier', 'LogisticRegression', 'LinearSVC', 'Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT", "HistGBDT"]#, 'Perceptron', 'MLP', 'SGD', 'SVC', 'NuSVC']

methods = ['RidgeClassifier', 'LogisticRegression', 'LinearSVC'] if methods_type == "linear" else ['Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT", "HistGBDT"]

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
              pca = PCA(n_components=0.95, svd_solver='full') #PCA(n_components=n_components)
              X_r = pca.fit_transform(X_s)
              print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
              print("X.shape reducted: ", X.shape)
            else:
              X_r = X_s.copy()
          
            for delta_i in delta:
              X_m = X_r.copy()
              Y_m = Y_.copy()
            
              X, y = evalClass(X_m, Y_m, delta_i)

              print("Delta:", delta_i, "X:", X.shape, "Y:", y.shape)
        
              log_fname = open(rof_dir + "results_"+str(int(delta_i*100))+".csv", "w")
              log_fname.write("method,auc_train,auc_test,acc_train, acc_test,f1_train,f1_test\n")

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
                
                scores_auc = []
                scores_f1 = []
                scores_acc = []
                for split in range(num_splits):
                
                  xtrain, xval, xtest, ytrain, yval, ytest = getClassSplit(X, y)
              
                  xtrain_val = np.concatenate([xtrain, xval])
                  ytrain_val = np.concatenate([ytrain, yval])
                  #print("X:", xtrain_val.shape, "Y:", ytrain_val.shape)
                
                  svr = getClassifier(m, 1e-15)

                  svr_model = svr.fit(xtrain_val, ytrain_val)
                  
                  [auc_train, _, _], f1_train, acc_train = getClassMetrics(svr_model, xtrain_val, ytrain_val)
                  [auc_test, precision, recall], f1_test, acc_test = getClassMetrics(svr_model, xtest, ytest)
                  
                  plt.figure()
                  plt.plot(recall, precision,'b')
                  plt.yticks(np.arange(0, 1.2, step=0.2))
                  plt.xticks(np.arange(0, 1.2, step=0.2))
                  plt.xlabel("Recall")
                  plt.ylabel("Presicion")
                  plt.title("Precision - Recall: "+str(round(auc_test,5)))
                  plt.grid(True)
                  plt.savefig(name_dir + "/charts/"+namefile_m+".png")
                  plt.clf()
                  plt.cla()
                  plt.close()
                  print("Split ",split,": Train:", auc_train, "Test:", auc_test)
                
                  scores_auc.append(auc_test)
                  scores_f1.append(f1_test)
                  scores_acc.append(acc_test)
                
                scores_auc = np.asarray(scores_auc)
                scores_f1 = np.asarray(scores_f1)
                scores_acc = np.asarray(scores_acc)
                
                print("Resume test: AUC:", scores_auc.mean(), "ACC:", scores_acc.mean(), "F1:",scores_f1.mean())
                
                log_fname.write(m+","+ str(round(auc_train,5)) +"," +str(round(scores_auc.mean(),5))+","+str(round(acc_train,5))+","+ str(round(scores_acc.mean(),5))+","+str(round(f1_train,5))+","+str(round(scores_f1.mean(),5))+"\n")
                
              log_fname.close()
                
