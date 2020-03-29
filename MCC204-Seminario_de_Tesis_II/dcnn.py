#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from scipy.io import savemat, loadmat

import numpy as np
from numpy import genfromtxt
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, KFold, train_test_split

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from utils import verifyFile, verifyDir
from utils.datasets import getScores, evalClass
from utils.downloaders import is_valid_image
from utils.networks import clearSession, getModel, verifyDevices, getPreprocess

from keras.models import Model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

years = ["2011", "2013", "2019"]

metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]

clearSession()
verifyDevices("gpu")

name_dir = "dcnn/"
verifyDir(name_dir)

delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

feature='vgg16'

## vgg part
for data_year in years:
  images_dir = "data/images/pp1/"+data_year+"/"
  for city in cities:
    acc =[]
    loss = []
    for metric in metrics:
      print("Using the model: ")
      curr_acc = []
      curr_loss = []
      
      file_log = open(name_dir+"vgg16_"+city+"_"+metric+"_"+data_year+"_.txt", "w")
      
      for delta_i in delta:
        #f = open(name_dir + "vgg16_"+city+"_"+metric+"_"+data_year+".csv", "w")
        #f.write("train,val,test\n")
        
        model = getModel(name_model='vgg16', include_top=True, weights='imagenet', freeze_layers=-2)
        cnn = model.layers[-2].output
        cnn = Dense(2, activation="softmax", name='famv_regressor')(cnn)
        
        dcnn_model = Model(inputs=model.input, outputs=cnn, name="famv-regressor")
        for layer in dcnn_model.layers:
          print(layer, layer.trainable)
        dcnn_model.summary()
        dcnn_model.compile(optimizer = "sgd", loss="categorical_crossentropy", metrics=['acc', 'mse'])
        
        checkpoint_file = data_year+"_"+city+"_"+metric+"_best.h5"
        checkpoints = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        print("preparing {}-{} data ...".format(city, metric))
        
        if feature == 'vgg16':
          pp_data = getScores(["placepulse_1", city, metric])
          images_list = pp_data[0]
          scores_list = pp_data[1]
          leng = len(images_list)
          X, y = [], []
          for i in range(0, leng):
            img = images_list[i]
            img_path = images_dir+str(img)+".jpg"
            if not is_valid_image(img_path):
              continue
            
            # loadImage
            orig, preprocess, img_name, img_dims, img_orig_dims = getPreprocess(img_path)
            X.append(preprocess)
            y.append(scores_list[i])
          X = np.asarray(X, dtype='float')
          #nsamples, nx, ny = X.shape
          #X = X.reshape((nsamples, nx*ny))
          y = np.asarray(y, dtype='float')
        else:
          mat = loadmat('features_'+data_year+'.mat')
          if feature == 'gist':
            gist = mat["gist_feature_matrix"]
            X = np.asarray(gist)
            print("X_gist.shape: ", X.shape)
          elif feature == 'fisher':
            fisher = mat["fisher_feature_matrix"]
            X = np.asarray(fisher)
            print("X_fisher.shape: ", X.shape)
          #image_names = mat["image_list"]
          scores = mat["scores"][0]
          y = np.asarray(scores, dtype='float')
        
        X, Y = evalClass(X, y, delta_i)
        X = np.asarray(X, dtype='float')
        Y = to_categorical(Y, num_classes=2)
        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.4)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, shuffle=True, test_size=0.5)
        
        print("X_train.shape:", X_train.shape, "X_val.shape:", X_val.shape, "X_test.shape:", X_test.shape)
        print("Y_train.shape:", Y_train.shape, "Y_val.shape:", Y_val.shape, "Y_test.shape:", Y_test.shape)
        
        dcnn_history = dcnn_model.fit(X_train, Y_train, batch_size=64, verbose=1, epochs=80, shuffle=True, validation_data=(X_val, Y_val), callbacks = [checkpoints])
        
        evaluations = dcnn_model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
        print("%s: %.2f%%" % (dcnn_model.metrics_names[0], evaluations[0]*100))
        print("%s: %.2f%%" % (dcnn_model.metrics_names[1], evaluations[1]*100))
        
        print("Plotting results ...")
        # plot with various axes scales
        plt.figure()

        plt.subplot(311)
        plt.plot(dcnn_history.history['acc'],'r')
        plt.plot(dcnn_history.history['val_acc'],'g')
        plt.yticks(np.arange(0, 1, step=0.05))
        plt.yscale('log')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training: Acc vs Epochs"+data_year+" Acc: "+str(round(evaluations[1]*100, 3)))
        plt.legend(['Train','Val'])
        plt.grid(True)

        plt.subplot(313)
        plt.plot(dcnn_history.history['loss'],'r')
        plt.plot(dcnn_history.history['val_loss'],'g')
        plt.yticks(np.arange(0, 1, step=0.05))
        plt.yscale('log')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training: Loss vs Epochs "+data_year+" Acc: "+str(round(evaluations[1]*100, 3)))
        plt.legend(['Train','Val'])
        plt.yscale('log')
        plt.grid(True)

        plt.savefig(name_dir+"vgg16_"+city+"_"+metric+"_"+data_year+"_.png")
        
        plt.clf()
        plt.cla()
        plt.close()
        
        loss_cities = evaluations[0]
        acc_cities = evaluations[1]
        curr_loss.append(loss_cities)
        curr_acc.append(acc_cities)
        
        file_log.write("\nDelta: "+ str(delta_i)+"\n")
        file_log.write("X-train: " + str(X_train.shape[0]) + " val: " + str(X_val.shape[0]) + " test: " + str(X_test.shape[0]))
        file_log.write("Y-train: " + str(Y_train.shape[0]) + " val: " + str(Y_val.shape[0]) + " test: " + str(Y_test.shape[0]))
        file_log.write("\n"+dcnn_model.metrics_names[0]+": "+ str(loss_cities*100)+ "\n")
        file_log.write(dcnn_model.metrics_names[1]+": "+ str(acc_cities*100)+ "\n")
        
        '''
        ypred = dcnn_model.predict(X_test, verbose=1)
        a, b = ypred.shape
        ypred = np.reshape(ypred, a*b)
        print(type(ypred), type(Y_test), Y_test[:5], ypred[:5])
        pearson_coef, _ = pearsonr(Y_test, ypred)
        print("Pearson correlation test is: ", pearson_coef)
        
        yval_pred = dcnn_model.predict(X_val, verbose=1)
        a, b = yval_pred.shape
        yval_pred = np.reshape(yval_pred, a*b)
        pearson_coef_val, _ = pearsonr(Y_val, yval_pred)
        print("Pearson correlation val is: ", pearson_coef_val)
        
        ytrain_pred = dcnn_model.predict(X_train, verbose=1)
        a, b = ytrain_pred.shape
        ytrain_pred = np.reshape(ytrain_pred, a*b)
        pearson_coef_train, _ = pearsonr(Y_train, ytrain_pred)
        print("Pearson correlation train is: ", pearson_coef_train)
        
        f.write(str(round(pearson_coef_train,5)) + ","+str(round(pearson_coef_val,5)) + "," +str(round(pearson_coef,5)) +"\n")
        f.close()
        '''
        clearSession(dcnn_model, "gpu")
      
      file_log.close()
      acc.append(curr_acc)
      loss.append(curr_loss)
      
    plt.figure()
    plt.plot(delta, acc[0],'r')
    #plt.axis([0.0, 0.55, 0.0, 1.0])
    plt.yticks(np.arange(0.5, 1, step=0.1))
    plt.xticks(np.arange(0, 0.55, step=0.05))
    plt.xlabel("Delta")
    plt.ylabel("Accuracy")
    plt.title("Testing "+metric+" trained in "+city)
    #plt.legend([cities[0], cities[1]])
    plt.grid(True)
    plt.savefig(name_dir+"vgg16_"+city+"_"+metric+"_"+data_year+"_acc_.png")
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.figure()
    plt.plot(delta, loss[0],'r')
    #plt.axis([0.0, 0.55, 0.0, 1.0])
    #plt.yticks(np.arange(0, 1, step=0.1))
    plt.xticks(np.arange(0, 0.55, step=0.05))
    plt.xlabel("Delta")
    plt.ylabel("Loss")
    plt.title("Testing")
    #plt.legend([cities[0], cities[1]])
    plt.grid(True)
    plt.savefig(name_dir+"vgg16_"+city+"_"+metric+"_"+data_year+"_loss_.png")
    plt.clf()
    plt.cla()
    plt.close()
