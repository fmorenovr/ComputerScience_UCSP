#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from numpy import genfromtxt

import json
from joblib import dump, load
from scipy.io import savemat, loadmat

from utils import verifyDir
from utils.datasets import getScores
from utils.datasets import evalClass
from utils.datasets import getClassSplit
from utils.networks import getLoss
from utils.networks import getModel
from utils.networks import getPreprocess
from utils.networks import getOptimizer
from utils.networks import freezeLayers, unFreezeLayers
from utils.networks import getClassMetric
from utils.networks import clearSession, verifyDevices, clearModel
from utils.networks import getLearningRate

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import PReLU, LeakyReLU
from keras.regularizers import l1, l2
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

clearSession()
verifyDevices("gpu")

batch_s, name_loss, opt_name, opt_lr = 128, "hinge", "sgd", 0.00099

years = ["2011", "2013", "2019"]
metrics = ["safety", "wealthy", "uniquely"]
cities = ["Boston", "New York City"]
delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

C = np.logspace(6, -6, 13)

alpha = np.logspace(6, -6, 13)

num_splits = 10

features_dir = "features/"

dir_to_save = "outputs/fine_tunned/classifier/"

results = {}

## vgg part
for data_year in years:
  name_dir_ = dir_to_save+str(data_year)+"/"
  verifyDir(name_dir_)
  images_dir = "data/images/pp1/"+data_year+"/"
  
  results[data_year] = {}
  
  for city in cities:
    
    results[data_year][city] = {}
    for metric in metrics:
      curr_acc = []
      curr_loss = []
      curr_auc = []
      curr_mse = []
      results[data_year][city][metric] = {}
      
      name_dir = name_dir_ + city+"/"+metric+"/fineTuning/"
      verifyDir(name_dir)
      
      file_log = open(name_dir+"log.txt", "w")
      
      print("preparing {}-{} data ...".format(city, metric))
      
      #X_ = load(features_dir+"X_"+data_year+"_"+city+"_"+metric+".joblib")
      #Y_ = load(features_dir+"y_"+data_year+"_"+city+"_"+metric+".joblib")
      
      X_ = genfromtxt(features_dir+"vgg16_gap"+"_X_"+data_year+"_"+city+"_"+metric+".csv", delimiter=',')
      Y_ = genfromtxt(features_dir+"vgg16_gap"+"_y_"+data_year+"_"+city+"_"+metric+".csv", delimiter=',')
      
      X_ = np.array(X_)
      #X_ = preprocess_input(X_)
      #nsamples, nx, ny = X.shape
      #X = X.reshape((nsamples, nx*ny))
      Y_ = np.array(Y_)
      
      for delta_i in delta:
        print("Preparing delta:", delta_i)
        X, Y = evalClass(X_, Y_, delta_i)

        print("Delta:", delta_i, "X:", X.shape, "Y:", Y.shape)
        
        best_loss = []
        best_acc = []
        best_mse = []
        best_auc = []
        
        delta_dir = name_dir+str(int(delta_i*100))
        verifyDir(delta_dir+"/")
        verifyDir(delta_dir+"/logs/")
        verifyDir(delta_dir+"/models/")
        verifyDir(delta_dir+"/charts/")
        verifyDir(delta_dir+"/data/")
          
        fname_max = open(delta_dir + "/"+"splits.csv", "w")
        fname_max.write("split,c,alpha,train,test\n")
        
        index = np.where(Y==0)[0]
        Y[index] = -1
        
        for split in range(num_splits):
          X_train, X_val, X_test, Y_train, Y_val, Y_test = getClassSplit(X, Y)
          
          print("X_train.shape:", X_train.shape, "X_val.shape:", X_val.shape, "X_test.shape:", X_test.shape)
          print("Y_train.shape:", Y_train.shape, "Y_val.shape:", Y_val.shape, "Y_test.shape:", Y_test.shape)
          
          print("Using the model: ")

          scores_val = []
          
          fname_split = open(delta_dir + "/logs/"+str(split)+".csv", "w")
          fname_split.write("c,alpha,train,val\n")
          
          for c in C:
            scores_val_alpha = []
            for alpha_val in alpha:
              model = Input(shape=(512,))
              cnn = Dense(1, activation="tanh")(model)

              dcnn_model = Model(inputs=model, outputs=cnn, name="famv-classifier")
              for layer in dcnn_model.layers:
                print(layer, layer.trainable)
              dcnn_model.summary()
            
              opt = getOptimizer(opt_name, opt_lr)
              loss = getLoss(dcnn_model, loss_name=name_loss, param_c=c, alpha=alpha_val)
              lr_scheduler = getLearningRate(alpha=alpha_val, decay_name="other")
              dcnn_model.compile(optimizer = opt, loss=loss, metrics=['acc', 'mse'])
              
              checkpoint_file = delta_dir+"/models/best_alpha_c.h5"
              checkpoints = ModelCheckpoint(checkpoint_file, monitor='acc', verbose=1, save_best_only=True, mode='max')
              
              lrate = LearningRateScheduler(lr_scheduler, verbose=1)
              
              earlyS = EarlyStopping(monitor='acc', patience=25, verbose=1, mode='max')
              
              dcnn_history = dcnn_model.fit(X_train, Y_train, batch_size=batch_s, verbose=1, epochs=60, validation_split=0.0, shuffle=True, callbacks = [checkpoints, earlyS, lrate])
              
              print("doing evaluations")
              
              model_to_predict = load_model(checkpoint_file, custom_objects={'loss': loss})
            
              evaluations = model_to_predict.evaluate(X_val, Y_val, batch_size=64, verbose=1)
              metric_loss = model_to_predict.metrics_names[0]
              loss_value = evaluations[0]
              metric_acc = model_to_predict.metrics_names[1]
              acc_value = evaluations[1]
              metric_mse = model_to_predict.metrics_names[2]
              mse_value = evaluations[2]
              print("%s: %.2f%%" % (metric_loss, loss_value*100))
              print("%s: %.2f%%" % (metric_acc, acc_value*100))
              print("%s: %.2f%%" % (metric_mse, mse_value*100))
              
              auc_train, _, _ = getClassMetric(model_to_predict, X_train, Y_train)
              auc_score, precision, recall = getClassMetric(model_to_predict, X_val, Y_val)
              print("%s: %.2f%%" % ("AUC", auc_score*100))
              
              fname_split.write(str(c)+","+str(alpha_val) +","+str(round(auc_train,5)) +"," +str(round(auc_score,5)) +"\n")

              scores_val_alpha.append(auc_score)
              
              clearSession()
              clearModel(model_to_predict)
              clearModel(dcnn_model)
              clearModel(cnn)
              clearModel(model)
              
            scores_val.append(scores_val_alpha)
            
          fname_split.close()
          
          aucs_vals = np.asarray(scores_val)
          #index_max_test = np.argmax()
          #c_val_max = C[index_max_test]
          vals = np.where(aucs_vals == np.amax(aucs_vals))
          c_max_index = vals[0][0]
          alpha_max_index = vals[1][0]
          
          c_val_max = C[c_max_index]
          alpha_val_max = alpha[alpha_max_index]
          
          #model = getModel(name_model='vgg16', include_top=True, weights='imagenet', freeze_layers=-1)
          #cnn = model.layers[-2].output
          #cnn = Dense(1, activation="tanh")(cnn)
          model = Input(shape=(512,))
          cnn = Dense(1, activation="tanh")(model)
          
          dcnn_model = Model(inputs=model, outputs=cnn, name="famv-classifier")
          for layer in dcnn_model.layers:
            print(layer, layer.trainable)
          dcnn_model.summary()
          
          opt = getOptimizer(opt_name, opt_lr)
          loss = getLoss(dcnn_model, loss_name=name_loss, param_c=c_val_max, alpha=alpha_val_max)
          lr_scheduler = getLearningRate(alpha=alpha_val_max, decay_name="other")
          dcnn_model.compile(optimizer = opt, loss=loss, metrics=['acc', 'mse'])
          
          checkpoint_file = delta_dir+"/models/best_"+str(split)+".h5"
          checkpoints = ModelCheckpoint(checkpoint_file, monitor='acc', verbose=1, save_best_only=True, mode='max')
          
          lrate = LearningRateScheduler(lr_scheduler, verbose=1)
          
          earlyS = EarlyStopping(monitor='acc', patience=25, verbose=1, mode='max')
          
          xtrain_split = shuffle(np.concatenate([X_train, X_val]), random_state=12)
          xtest_split = shuffle(X_test.copy(), random_state=35)

          ytrain_split = shuffle(np.concatenate([Y_train, Y_val]), random_state=12)
          ytest_split = shuffle(Y_test.copy(), random_state=35)
          
          dcnn_history = dcnn_model.fit(xtrain_split, ytrain_split, batch_size=batch_s, verbose=1, epochs=60, validation_split=0.0, shuffle=True, callbacks = [checkpoints, earlyS, lrate])
              
          print("doing evaluations")
          
          model_to_predict = load_model(checkpoint_file, custom_objects={'loss': loss})
          
          evaluations = model_to_predict.evaluate(xtest_split, ytest_split, batch_size=64, verbose=1)
          metric_loss = model_to_predict.metrics_names[0]
          loss_value = evaluations[0]
          metric_acc = model_to_predict.metrics_names[1]
          acc_value = evaluations[1]
          metric_mse = model_to_predict.metrics_names[2]
          mse_value = evaluations[2]
          print("%s: %.2f%%" % (metric_loss, loss_value*100))
          print("%s: %.2f%%" % (metric_acc, acc_value*100))
          print("%s: %.2f%%" % (metric_mse, mse_value*100))
          
          auc_train, _, _ = getClassMetric(model_to_predict, xtrain_split, ytrain_split)
          auc_score, precision, recall = getClassMetric(model_to_predict, xtest_split, ytest_split)
          print("%s: %.2f%%" % ("AUC", auc_score*100))
          
          clearModel(model_to_predict)
          clearModel(cnn)
          clearModel(model)
          clearModel(dcnn_model)
          clearSession()
          
          if auc_score > 0.89:
            tostore=dict(zip(['X_train_split', 'X_test_split', 'y_train_split', 'y_test_split', 'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest', 'c', 'alpha'], [xtrain_split, xtest_split, ytrain_split, ytest_split, X_train, X_val, X_test, Y_train, Y_val, Y_test, c_val_max, alpha_val_max]))
            dump(tostore, delta_dir + "/data/"+str(split)+'.joblib')

          print("Plotting results ...")
          # plot with various axes scales
          plt.figure()

          plt.subplot(311)
          plt.plot(dcnn_history.history['acc'],'r')
          #plt.plot(dcnn_history.history['val_acc'],'g')
          plt.yticks(np.arange(0, 1, step=0.05))
          plt.yscale('log')
          plt.xlabel("Epochs")
          plt.ylabel("Accuracy")
          plt.title("Training: Acc vs Epochs"+data_year+" Acc: "+str(round(acc_value, 3)))
          plt.legend(['Train','Val'])
          plt.grid(True)

          plt.subplot(313)
          plt.plot(dcnn_history.history['loss'],'r')
          #plt.plot(dcnn_history.history['val_loss'],'g')
          plt.yticks(np.arange(0, 1, step=0.05))
          plt.yscale('log')
          plt.xlabel("Epochs")
          plt.ylabel("Loss")
          plt.title("Training: Loss vs Epochs "+data_year+" Loss: "+str(round(loss_value, 3)))
          plt.legend(['Train','Val'])
          plt.yscale('log')
          plt.grid(True)
          plt.savefig(delta_dir+"/charts/acc-loss_"+str(split)+".png")
          plt.clf()
          plt.cla()
          plt.close()
          
          plt.figure()
          plt.plot(recall, precision,'b')
          plt.yticks(np.arange(0, 1.2, step=0.2))
          plt.xticks(np.arange(0, 1.2, step=0.2))
          plt.xlabel("Recall")
          plt.ylabel("Presicion")
          plt.title("Precision - Recall, AUC: "+str(auc_score))
          plt.grid(True)
          plt.savefig(delta_dir+"/charts/precision-recall_"+str(split)+".png")
          plt.clf()
          plt.cla()
          plt.close()
          
          best_loss.append(loss_value)
          best_acc.append(acc_value)
          best_mse.append(mse_value)
          best_auc.append(auc_score)
          
          fname_max.write(str(split)+","+str(c_val_max)+","+str(alpha_val_max)+","+ str(round(auc_train,5)) +"," +str(round(auc_score,5)) +"\n")
          
          print("Resume, Train:", auc_train, "Test:", auc_score)
          
        fname_max.close()
        
        curr_loss.append(np.asarray(best_loss).mean())
        curr_acc.append(np.asarray(best_acc).mean())
        curr_mse.append(np.asarray(best_mse).mean())
        curr_auc.append(np.asarray(best_auc).mean())
        
        file_log.write("\nDelta: "+ str(delta_i)+"\n")
        file_log.write("X-train: " + str(X_train.shape[0]) + " val: " + str(X_val.shape[0]) + " test: " + str(X_test.shape[0])+"\n")
        file_log.write("Y-train: " + str(Y_train.shape[0]) + " val: " + str(Y_val.shape[0]) + " test: " + str(Y_test.shape[0])+"\n")
        file_log.write(metric_loss+": "+ str(np.asarray(best_loss).mean())+ "\n")
        file_log.write(metric_acc+": "+ str(np.asarray(best_acc).mean())+ "\n")
        file_log.write(metric_mse+": "+ str(np.asarray(best_mse).mean())+ "\n")
        file_log.write("AUC: "+ str(np.asarray(best_auc).mean())+ "\n")
        
      results[data_year][city][metric] = {'loss': curr_loss,
                                          'mse': curr_mse,
                                          'auc': curr_auc,
                                          'acc': curr_acc}
      
      with open(name_dir + "results_summary.json", 'w') as outfile:
        json.dump(results[data_year][city][metric], outfile)
      
for data_year in years:
  for city in cities:
    for metric in metrics:
      plt.figure()
      plt.plot(delta, results[data_year][city][metric]["acc"],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("Accuracy")
      plt.title("Testing "+metric+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metric+"/fineTuning/acc.png")
      plt.clf()
      plt.cla()
      plt.close()
      
      plt.figure()
      plt.plot(delta, results[data_year][city][metric]["loss"],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      #plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("Loss (Binarycross)")
      plt.title("Testing "+metric+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metric+"/fineTuning/loss.png")
      plt.clf()
      plt.cla()
      plt.close()
      
      plt.figure()
      plt.plot(delta, results[data_year][city][metric]["mse"],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      #plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("MSE")
      plt.title("Testing "+metric+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metric+"/fineTuning/mse.png")
      plt.clf()
      plt.cla()
      plt.close()
      
      plt.figure()
      plt.plot(delta, results[data_year][city][metric]["auc"],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("AUC")
      plt.title("Testing "+metric+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metric+"/fineTuning/auc.png")
      plt.clf()
      plt.cla()
      plt.close()
      
