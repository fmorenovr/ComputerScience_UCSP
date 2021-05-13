#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from joblib import dump, load

from utils import verifyDir
from utils.datasets import evalClass, getClassSplit, getScores
from utils.libsvm import getClassifier, getClassMetrics

from matplotlib.colors import Normalize

# CLASSIFICATION

input_dir = "features/"

output_dir_ = "to_test/"

psp_df = pd.read_csv(input_dir+'segmented_df.csv')
#print(psp_df)

gap_df = pd.read_csv(input_dir+"vgg16_gap_2011_Boston_safety.csv")
#print(gap_df)

gap_places = pd.read_csv(input_dir+"vgg16_gap_places_2011_Boston_safety.csv")

#import numpy as np
#print(np.sum(gap_df["y"].values == psp_df["y"].values))
#print(np.sum(gap_df["y"].values == psp_df["y"].values))

dataset_plus = pd.merge(gap_df, psp_df, on='ID', how="inner")
#import numpy as np
#print(np.sum(dataset_plus.iloc[:, 513].values == dataset_plus.iloc[:, -1].values))
dataset_plus.drop(columns=["y_x"], inplace=True)
dataset_plus.rename(columns={'y_y': 'y'}, inplace=True)
#dataset_plus.iloc[:, 513].values
dataset_plus.sort_values("y", ascending=False, inplace=True)

dataset_plus = pd.merge(dataset_plus, gap_places, on="ID", how="inner")
dataset_plus.drop(columns=["y_x"], inplace=True)
dataset_plus.rename(columns={'y_y': 'y'}, inplace=True)

#print(X_df)
#print(y_df)

C = np.logspace(5, -5, 11)

num_splits = 10

delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'red', 'blue']
#methods = ['RidgeClassifier', 'Perceptron', 'LogisticRegression', 'LinearSVC', 'MLP', 'SGD']#, 'SVC', 'NuSVC']
methods = ['RidgeClassifier', 'LogisticRegression', 'LinearSVC']#, "SVC"]
#random_methods = ['Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT"]#, "HistGBDT"]
random_methods = []
colors = colors[:len(methods)]

auc_results = {'length': 0,
               'Perceptron': [],
               'RidgeClassifier': [],
               'LogisticRegression': [],
               'LinearSVC': [],
               'MLP': [],
               'SVC': [],
               'NuSVC': [],
               'SGD': [],
               'Bayes':[],
               'Tree':[]}

verifyDir(output_dir_)

class MidpointNormalize(Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    Normalize.__init__(self, vmin, vmax, clip)

  def __call__(self, value, clip=None):
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y))

for delta_i in delta:
  rbf_map={}
  for type_feature in ["psp", "gap", "gap_places"]:
    if type_feature == "psp":
      X_df = dataset_plus.iloc[:,513:-513].copy()
      X_df["ID"] = dataset_plus["ID"].values
      cols = ['ID']  + [col for col in X_df if col != 'ID']
      X_df = X_df[cols]
      y_df = dataset_plus.iloc[:,[0,-1]].copy()
    elif type_feature == "gap":
      X_df = dataset_plus.iloc[:,:513].copy()
      #X_df["ID"] = dataset_plus["ID"].values
      #cols = ['ID']  + [col for col in X_df if col != 'ID']
      #X_df = X_df[cols]
      y_df = dataset_plus.iloc[:,[0,-1]].copy()
    elif type_feature == "gap_places":
      X_df = dataset_plus.iloc[:,-513:-1].copy()
      X_df["ID"] = dataset_plus["ID"].values
      cols = ['ID']  + [col for col in X_df if col != 'ID']
      X_df = X_df[cols]
      y_df = dataset_plus.iloc[:,[0,-1]].copy()
    #print(X_df)
    #print(y_df)
    #continue
  
    output_dir = output_dir_ + str(int(delta_i*100))+"/"
    verifyDir(output_dir)
    #X_m = X_df.to_numpy(copy=True)
    #Y_m = y_df.to_numpy(copy=True)
    
    #X, y = evalClass(X_m, Y_m, delta_i)
    slen = len(y_df)
    val = round(delta_i*slen)
    
    scores_df = y_df.copy()
    scores_df["class"] = scores_df['y']
    scores_df['class'].iloc[:val+1]=1
    scores_df['class'].iloc[slen-val:] = 0
    
    X_pos = X_df.iloc[:val+1]
    X_neg = X_df.iloc[slen-val:]
    y_pos = scores_df.iloc[:val+1]
    y_neg = scores_df.iloc[slen-val:]
    
    from sklearn.model_selection import train_test_split
    
    xtrain_pos, xtest_pos, ytrain_pos, ytest_pos = train_test_split(X_pos, y_pos, shuffle=True, test_size = 0.25, random_state=35)
    xtrain_neg, xtest_neg, ytrain_neg, ytest_neg = train_test_split(X_neg, y_neg, shuffle=True, test_size = 0.25, random_state=35)
    
    xtrain_val = pd.concat([xtrain_pos, xtrain_neg])
    xtest = pd.concat([xtest_pos, xtest_neg])
    
    ytrain_val = pd.concat([ytrain_pos, ytrain_neg])
    ytest = pd.concat([ytest_pos, ytest_neg])
    
    tostore=dict(zip(['xtrain_val', 'ytrain_val', 'xtest', 'ytest'], [xtrain_val, ytrain_val, xtest, ytest]))
    dump(tostore, output_dir + 'dataset_'+type_feature+'.joblib')
    
    xtrain_val = xtrain_val.iloc[:,1:].to_numpy(copy=True)
    xtest = xtest.iloc[:,1:].to_numpy(copy=True)
    ytrain_val = ytrain_val.iloc[:,-1].to_numpy(copy=True)
    ytest = ytest.iloc[:,-1].to_numpy(copy=True)
    #exit()
    
    from sklearn.utils import shuffle
    from sklearn.model_selection import StratifiedShuffleSplit
    
    xtest, ytest = shuffle(xtest, ytest, random_state=25)
    
    #print("Delta:", delta_i, "X:", X.shape, "Y:", y.shape)
    
    #xtrain, xval, xtest, ytrain, yval, ytest = getClassSplit(X, y, random_state=35)
    #xtrain_val = np.concatenate([xtrain, xval])
    #ytrain_val = np.concatenate([ytrain, yval])
    
    kf = KFold(n_splits=num_splits)
    sf = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.25, random_state=15)
    
    log_fname = open(output_dir + "results_"+type_feature+".csv", "w")
    log_fname.write("method,c,auc_train,auc_test,acc_train, acc_test,f1_train,f1_test\n")
    
    print("LINEAR METHODS")
    
    for m in methods:
      print("Method:", m)
      #print("X:", xtrain_val.shape, "Y:", ytrain_val.shape)
      
      scores_global_val =[]
      scores_global_train = []
      split= 0
      for i_train, i_val in sf.split(xtrain_val, ytrain_val):
        xtrain, xval, ytrain, yval = xtrain_val[i_train], xtrain_val[i_val], ytrain_val[i_train], ytrain_val[i_val]

        #print("Split:", split)
        #print("xtrain:", xtrain.shape, "xval:", xval.shape, "xtest:", xtest.shape)
        #print("ytrain:", ytrain.shape, "yval:", yval.shape, "ytest:", ytest.shape)
       
        scores_val = []
        scores_train = []
        
        for c in C:
          #print("Evaluating with method", m, "with c=",c)
          svr = getClassifier(m, c)
          svr_model = svr.fit(xtrain, ytrain)
          
          # R, mse, mrsq
          [auc_train, _, _], _,_ = getClassMetrics(svr_model, xtrain, ytrain)
          [auc_val, _, _], _,_ = getClassMetrics(svr_model, xval, yval)

          scores_val.append(auc_val)
          scores_train.append(auc_train)
          
          #print("train:", auc_train, "val:", auc_val)
          
        scores_global_val.append(scores_val)
        scores_global_train.append(scores_train)
        print(split, c, np.amax(scores_global_val))
        split=split+1
        
      scores_global_val = np.asarray(scores_global_val)
      scores_means = np.nanmean(scores_global_val, axis=0)
      
      scores_global_train = np.asarray(scores_global_train)
      scores_means_train = np.nanmean(scores_global_train, axis=0)
      
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
      
      print("Resume: Train:", auc_train, "Test:", auc_test)
                  
      log_fname.write(m+","+str(c_val)+","+ str(round(auc_train,5)) +"," +str(round(auc_test,5))+","+str(round(acc_train,5))+","+ str(round(acc_test,5))+","+str(round(f1_train,5))+","+str(round(f1_test,5))+"\n")
      
      dump(svr_model, output_dir + m+'_'+type_feature+'.joblib')
    ''''
    print("RANDOM METHODS")
    
    for m in random_methods:
      print("Method:", m)
      print("X:", xtrain_val.shape, "Y:", ytrain_val.shape)
      
      scores_auc = []
      scores_f1 = []
      scores_acc = []
      
      for split in range(num_splits):
        xtrain, xval, xtest, ytrain, yval, ytest = getClassSplit(X, y, random_state=35)
        
        xtrain_val = np.concatenate([xtrain, xval])
        ytrain_val = np.concatenate([ytrain, yval])
        #print("X:", xtrain_val.shape, "Y:", ytrain_val.shape)
      
        svr = getClassifier(m, 0)

        svr_model = svr.fit(xtrain_val, ytrain_val)
        
        [auc_train, _, _], f1_train, acc_train = getClassMetrics(svr_model, xtrain_val, ytrain_val)
        [auc_test, precision, recall], f1_test, acc_test = getClassMetrics(svr_model, xtest, ytest)
      
        scores_auc.append(auc_test)
        scores_f1.append(f1_test)
        scores_acc.append(acc_test)
        
        print("Split:", split, "auc train:", auc_train, "auc_test:", auc_test)
      
      scores_auc = np.asarray(scores_auc)
      scores_f1 = np.asarray(scores_f1)
      scores_acc = np.asarray(scores_acc)
      
      print("Resume: Train:", auc_train, "Test:", scores_auc.mean())
       
      log_fname.write(m+",no,"+ str(round(auc_train,5)) +"," +str(round(scores_auc.mean(),5))+","+str(round(acc_train,5))+","+ str(round(scores_acc.mean(),5))+","+str(round(f1_train,5))+","+str(round(scores_f1.mean(),5))+"\n")
      
      dump(svr_model, output_dir + m+'.joblib')
    '''
    log_fname.close()
      
    from sklearn.svm import SVC
    import math
    
    C_range = np.logspace(5, -5, 11)
    G_range = np.logspace(5, -5, 11)
    
    scores_global_val =[]
    split= 0
    log_fname = open(output_dir + "results_"+type_feature+".csv", "a")
    for i_train, i_val in sf.split(xtrain_val, ytrain_val):
      xtrain, xval, ytrain, yval = xtrain_val[i_train], xtrain_val[i_val], ytrain_val[i_train], ytrain_val[i_val]
      
      scores_val_c = []
      for c in C_range:
        scores_val_g = []
        for g in G_range:
          #print("Evaluating with method", m, "with c=",c)
          svr =  SVC(kernel="rbf", C=c, gamma=g)
          svr_model = svr.fit(xtrain, ytrain)
          
          # R, mse, mrsq
          [auc_train, _, _], _,_ = getClassMetrics(svr_model, xtrain, ytrain)
          [auc_val, _, _], _,_ = getClassMetrics(svr_model, xval, yval)
          
          ppar = math.isnan(auc_val)
          if ppar:
            auc_val = 0
          
          scores_val_g.append(auc_val)
        scores_val_c.append(scores_val_g)
        
      scores_global_val.append(scores_val_c)
      print(split, c, g, np.amax(scores_val_c))
      split=split+1
    
    scores_test = np.asarray(scores_global_val) # matrix: c x g
    #print(scores_test.shape)
    scores_means = np.mean(scores_test, axis=(0))
    #print(scores_means)
    score_max = np.amax(scores_means)
    #print(score_max)
    #print(np.where(scores_means == score_max))
    c_max, g_max = np.where(scores_means == score_max)
    #print(c_max, g_max)
    c_max, g_max = c_max[0], g_max[0]
    svr = SVC(kernel="rbf", C=C_range[c_max], gamma=G_range[g_max])
    svr_model = svr.fit(xtrain_val, ytrain_val)
        
    [auc_train, _, _], f1_train, acc_train = getClassMetrics(svr_model, xtrain_val, ytrain_val)
    [auc_test, precision, recall], f1_test, acc_test = getClassMetrics(svr_model, xtest, ytest)
    
    print("Resume: Train:", auc_train, "Test:", auc_test)
               
    log_fname.write("method,c,g,auc_train,auc_test,acc_train, acc_test,f1_train,f1_test\n") 
    log_fname.write("RBF,"+str(C_range[c_max])+","+str(G_range[g_max])+","+ str(round(auc_train,5)) +"," +str(round(auc_test,5))+","+str(round(acc_train,5))+","+ str(round(acc_test,5))+","+str(round(f1_train,5))+","+str(round(f1_test,5))+"\n")
    
    dump(svr_model, output_dir +'RBF_'+type_feature+'.joblib')
    
    rbf_map[type_feature] = scores_means
    
    '''
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_means, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(G_range)), G_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy test')#; PCA components='+str(n_features))
    plt.savefig(output_dir+type_feature+"_cmap.png")
    plt.clf()
    plt.cla()
    plt.close()
    '''
    
    log_fname.close()
    
  fig = plt.figure(figsize=(16, 8))
  plt.subplot(1, 3, 1)
  #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(rbf_map['psp'], interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
  plt.xlabel('gamma')
  plt.ylabel('C')
  plt.colorbar()
  plt.xticks(np.arange(len(G_range)), G_range, rotation=45)
  plt.yticks(np.arange(len(C_range)), C_range)
  plt.title('Accuracy test PSP')#; PCA components='+str(n_features))

  plt.subplot(1, 3, 2)
  #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(rbf_map['gap'], interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
  plt.xlabel('gamma')
  plt.ylabel('C')
  plt.colorbar()
  plt.xticks(np.arange(len(G_range)), G_range, rotation=45)
  plt.yticks(np.arange(len(C_range)), C_range)
  plt.title('Accuracy test GAP')#; PCA components='+str(n_features))
  
  plt.subplot(1, 3, 3)
  #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(rbf_map['gap_places'], interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
  plt.xlabel('gamma')
  plt.ylabel('C')
  plt.colorbar()
  plt.xticks(np.arange(len(G_range)), G_range, rotation=45)
  plt.yticks(np.arange(len(C_range)), C_range)
  plt.title('Accuracy test GAP Places')#; PCA components='+str(n_features))

  plt.savefig(output_dir+"rbf_cmap.png")
  plt.clf()
  plt.cla()
  plt.close()

  '''  
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import make_scorer, accuracy_score
  parameters = {'kernel':('linear', 'rbf'), 'C':np.logspace(4, -4, 9), "gamma":np.logspace(4, -4, 9)}
  cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=35)
  scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
  
  #cv_results = cross_validate(SVC, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
  clf = GridSearchCV(estimator=SVC(), param_grid=parameters, scoring=scoring, n_jobs=-1, refit='AUC', return_train_score=True, verbose=10)
  
  clf_model = clf.fit(X, y)
  
  dump(clf.cv_results_, output_dir +'RBF_result.joblib')
  
  print(clf.cv_results_)
  '''
