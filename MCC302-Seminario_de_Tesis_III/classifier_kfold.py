#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import json
import numpy as np

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
output_dir = "outputs/kfold/classifier_"+methods_type+"/"

features = ['gist', "vgg16_places", 'vgg16_gap_places', 'vgg16', 'vgg16_gap']#, 'fisher']

methods = ['RidgeClassifier', 'LogisticRegression', 'LinearSVC', 'Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT", "HistGBDT"]#, 'Perceptron', 'MLP', 'SGD', 'SVC', 'NuSVC']

methods = ['RidgeClassifier', 'LogisticRegression', 'LinearSVC'] if methods_type == "linear" else ['Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT"]

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
        
              xtrain, xval, xtest, ytrain, yval, ytest = getClassSplit(X, y)
              
              xtrain_val = np.concatenate([xtrain, xval])
              ytrain_val = np.concatenate([ytrain, yval])
              
              kf = KFold(n_splits=num_splits)

              log_fname = open(rof_dir + "results_"+str(int(delta_i*100))+".csv", "w")
              log_fname.write("method,c,auc_train,auc_test,acc_train, acc_test,f1_train,f1_test\n")

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
                    svr = getClassifier(m, c)
                    svr_model = svr.fit(xtrain, ytrain)
                    
                    # R, mse, mrsq
                    [auc_train, _, _], _,_ = getClassMetrics(svr_model, xtrain, ytrain)
                    [auc_val, _, _], _,_ = getClassMetrics(svr_model, xval, yval)

                    scores_val.append(auc_val)
                    scores_train.append(auc_train)
                    
                    print("train:", auc_train, "val:", auc_val)
                    
                    fname_split.write(str(c) + ","+str(round(auc_train,5)) +"," +str(round(auc_val,5)) +"\n")
                    
                  fname_split.close()
                  scores_global_val.append(scores_val)
                  scores_global_train.append(scores_train)
                  split=split+1
                
                scores_global_val = np.asarray(scores_global_val)
                scores_means = np.nanmean(scores_global_val, axis=0)
                
                scores_global_train = np.asarray(scores_global_train)
                scores_means_train = np.nanmean(scores_global_train, axis=0)
                
                fname_max = open(name_dir + "/means.csv", "w")
                fname_max.write("c,train,val\n")
                for indx in range(len(scores_means)):
                  fname_max.write(str(C[indx])+","+ str(round(scores_means_train[indx],5)) +"," +str(round(scores_means[indx],5)) +"\n")
                  
                fname_max.close()
                
                nan_array = np.isnan(scores_means)
                not_nan_array = ~ nan_array
                scores_means = scores_means[not_nan_array]
                C_ = C[not_nan_array]
                
                index_max = np.argmax(scores_means)
                c_val = C_[index_max]
                
                svr = getClassifier(m, c)

                svr_model = svr.fit(xtrain_val, ytrain_val)
                  
                [auc_train, _, _], f1_train, acc_train = getClassMetrics(svr_model, xtrain_val, ytrain_val)
                [auc_test, precision, recall], f1_test, acc_test = getClassMetrics(svr_model, xtest, ytest)

                plt.figure()
                plt.plot(recall, precision,'b')
                plt.yticks(np.arange(0, 1.2, step=0.2))
                plt.xticks(np.arange(0, 1.2, step=0.2))
                plt.xlabel("Recall")
                plt.ylabel("Presicion")
                plt.title("Precision - Recall: "+str(round(auc_test,5))+", c="+str(c_val))
                plt.grid(True)
                plt.savefig(name_dir + "/charts/"+namefile_m+".png")
                plt.clf()
                plt.cla()
                plt.close()
                print("Resume: Train:", auc_train, "Test:", auc_test)
                
                log_fname.write(m+","+str(c_val)+","+ str(round(auc_train,5)) +"," +str(round(auc_test,5))+","+str(round(acc_train,5))+","+ str(round(acc_test,5))+","+str(round(f1_train,5))+","+str(round(f1_test,5))+"\n")
                dump(svr_model, name_dir+"/models/" + m+'.joblib')
                
              log_fname.close()
                
