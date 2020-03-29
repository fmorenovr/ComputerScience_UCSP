#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np
from numpy import genfromtxt
import pandas as pd
import random

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from utils import verifyFile, verifyDir

metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]
cities_test = ["Boston", "New York City"]
C = np.logspace(7, -7, 15)

threshold = 0.015
n_components = 2

#name_dir = "LASSO/"
#verifyDir(name_dir)

standard = True

for city in cities:
  for metric in metrics:
    print("preparing {}-{} data ...".format(city, metric))
    X = genfromtxt("X_"+city+".csv", delimiter=',')
    y = genfromtxt("y_"+city+".csv", delimiter=',')
    #y = y.astype(int)
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)

    feat_cols = [ 'pixel_'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['y_int'] = y.astype(int)
    df['label'] = df['y_int'].apply(lambda i: str(i))
    num_colors = len(np.unique(df['y_int']))
    print('Size of the dataframe: {}'.format(df.shape))

    '''
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(X,y)
    importances = model.feature_importances_
    indices = np.nonzero(importances>0.01)[0]
    #indices = np.argsort(importances)[-20:] #top 20 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feat_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    X_reduct = X[:,indices]
    '''
    
    if standard:
      scaler = StandardScaler()
      X = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95, svd_solver='full')#PCA(n_components=n_components)
    #ica = FastICA()
    #tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    X_reduct = pca.fit_transform(X)
    #X_reduct = X
    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print("X.shape reducted: ", X_reduct.shape)
    dimensions = range(0, X_reduct.shape[1])
    index_0 = random.choice(dimensions)
    index_1 = random.choice(dimensions)
    
    while index_0 == index_1:
      index_1 = random.choice(dimensions)
    
    print("selected dimensions: ", index_0, index_1)
    
    y_reduct = y/10
    
    xtrain, xtest, ytrain, ytest = train_test_split(X_reduct, y_reduct, test_size = 0.5)
    
    X_reduct_min = int(np.min(X_reduct[:,index_0])) - 1
    X_reduct_max = int(np.max(X_reduct[:,index_0])) + 2
    X_reduct_0 = range(X_reduct_min, X_reduct_max)
    #print(np.min(X_reduct[:,index_0]), np.max(X_reduct[:,index_0]), X_reduct_min, X_reduct_max)
    X_reduct_min = int(np.min(X_reduct[:,index_1])) - 1
    X_reduct_max = int(np.max(X_reduct[:,index_1])) + 2
    X_reduct_1 = range(X_reduct_min, X_reduct_max)
    #print(np.min(X_reduct[:,index_1]), np.max(X_reduct[:,index_1]), X_reduct_min, X_reduct_max)
    xx, yy = np.meshgrid(X_reduct_0, X_reduct_1)
    
    f = open("test.csv", "w")
    f.write("c,train,test\n")
    
    for c in C:
      print("Evaluating with c=",c)
      #svr = LinearSVR(C=c)
      svr = LinearRegression()
      #svr = Lasso(alpha=c)
      #svr = Ridge(alpha=c)
      svr_model = svr.fit(xtrain, ytrain)
      
      z = svr_model.coef_[index_0]*xx + svr_model.coef_[index_1]*yy + svr_model.intercept_
      
      ypred = svr_model.predict(xtest)
      pearson_coef, _ = pearsonr(ytest, ypred)
      print("Pearson correlation test is: ", pearson_coef)
      print("R2 score is: ", svr_model.score(xtest, ytest), "\n")
      
      y_orig = np.asarray([[ytest[i], ytest[i]] for i in range(len(ytest))])
      y_predorig = np.asarray([[ytest[i], ypred[i]] for i in range(len(ytest))])
      y_dis_pred = np.asarray([np.linalg.norm(y_orig[i] - y_predorig[i]) for i in range(len(y_orig))])
      
      #print("Values bigger than 10 =", y_dis[y_dis<threshold])
      indexes_pred = np.nonzero(y_dis_pred<threshold)
      
      ytrain_pred = svr_model.predict(xtrain)      
      pearson_coef_train, _ = pearsonr(ytrain, ytrain_pred)
      print("Pearson correlation train is: ", pearson_coef_train)
      print("R2 score is: ", svr_model.score(xtrain, ytrain), "\n")
      
      f.write(str(c) + "," + str(round(pearson_coef_train,5)) + "," +str(round(pearson_coef,5)) +"\n")
      
      y_orig = np.asarray([[ytrain[i], ytrain[i]] for i in range(len(ytrain))])
      y_predorig = np.asarray([[ytrain[i], ytrain_pred[i]] for i in range(len(ytrain))])
      y_dis_train = np.asarray([np.linalg.norm(y_orig[i] - y_predorig[i]) for i in range(len(y_orig))])
      
      #print("Values bigger than 10 =", y_dis[y_dis<threshold])
      indexes_train = np.nonzero(y_dis_train<threshold)
      
      plt.figure(figsize=(16,16))
      ax = plt.subplot(241, projection='3d')
      ax.scatter(
        xs=xtrain[:,index_0],
        ys=xtrain[:,index_1],
        zs=ytrain_pred,
        c=ytrain_pred,
        cmap='tab10'
      )
      ax.plot_surface(xx, yy, z, alpha=0.2)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('ytrain pred')
      ax.set_title("ytrain: C="+str(c)+" pearson="+str(round(pearson_coef_train,5)))
      
      ax = plt.subplot(2, 4, 2)
      ax.plot(ytrain, ytrain,'r')
      ax.plot(ytrain, ytrain_pred,'bo', alpha=0.1)
      #ax.set_yticks(np.arange(0, 1, step=0.05))
      #ax.set_xticks(np.arange(0, 1, step=0.05))
      ax.set_title("ytrain: C="+str(c)+" pearson="+str(round(pearson_coef_train,5))+" total= "+str(len(ytrain)))
      ax.legend(["valor real", "valor predicho"])
      ax.grid(True)
      
      ax = plt.subplot(2, 4, 3)
      ax.plot(ytrain[indexes_train], ytrain[indexes_train],'r')
      ax.plot(ytrain[indexes_train], ytrain_pred[indexes_train],'bo', alpha=0.1)
      #ax.set_yticks(np.arange(0, 1, step=0.05))
      #ax.set_xticks(np.arange(0, 1, step=0.05))
      ax.set_title("ytrain: threshold="+str(threshold)+" num="+str(len(indexes_train[0])))
      ax.legend(["valor real", "valor predicho"])
      ax.grid(True)
      
      ax = plt.subplot(244, projection='3d')
      ax.scatter(
        xs=xtrain[indexes_train, index_0],
        ys=xtrain[indexes_train, index_1],
        zs=ytrain_pred[indexes_train],
        c=ytrain_pred[indexes_train],
        cmap='tab10'
      )
      ax.plot_surface(xx, yy, z, alpha=0.2)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('ytrain pred')
      ax.set_title("ytrain predictions points")
      
      ax = plt.subplot(245,  projection='3d')
      ax.scatter(
        xs=xtest[:,index_0],
        ys=xtest[:,index_1],
        zs=ypred,
        c=ypred,
        cmap='tab10'
      )
      ax.plot_surface(xx, yy, z, alpha=0.2)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('ytest pred')
      ax.set_title("ytest: C="+str(c)+" pearson="+str(round(pearson_coef,5)))
      
      ax = plt.subplot(2, 4, 6)
      ax.plot(ytest, ytest,'r')
      ax.plot(ytest, ypred,'bo', alpha=0.1)
      #ax.set_yticks(np.arange(0, 1, step=0.05))
      #ax.set_xticks(np.arange(0, 1, step=0.05))
      ax.set_title("ytest: C="+str(c)+" pearson="+str(round(pearson_coef,5))+" total= "+str(len(ytest)))
      ax.legend(["valor real", "valor predicho"])
      ax.grid(True)
      
      ax = plt.subplot(2, 4, 7)
      ax.plot(ytest[indexes_pred], ytest[indexes_pred],'r')
      ax.plot(ytest[indexes_pred], ypred[indexes_pred],'bo', alpha=0.1)
      #ax.set_yticks(np.arange(0, 1, step=0.05))
      #ax.set_xticks(np.arange(0, 1, step=0.05))
      ax.set_title("ytest: threshold="+str(threshold)+" num="+str(len(indexes_pred[0])))
      ax.legend(["valor real", "valor predicho"])
      ax.grid(True)
      
      ax = plt.subplot(248,  projection='3d')
      ax.scatter(
        xs=xtest[indexes_pred, index_0],
        ys=xtest[indexes_pred, index_1],
        zs=ypred[indexes_pred],
        c=ypred[indexes_pred],
        cmap='tab10'
      )
      ax.plot_surface(xx, yy, z, alpha=0.2)
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_zlabel('ytest pred')
      ax.set_title("ytest predictions points")
      
      #plt.show()

      # plt.savefig(name_dir+str(c)+"_"+city+"_"+metric+".png")
      # plt.clf()
      # plt.cla()
      # plt.close()
