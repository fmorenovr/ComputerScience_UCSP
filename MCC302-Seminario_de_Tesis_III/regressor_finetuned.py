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
from utils.datasets import getRegressSplit
from utils.networks import getLoss
from utils.networks import getModel
from utils.networks import getPreprocess
from utils.networks import getOptimizer
from utils.networks import freezeLayers, unFreezeLayers
from utils.networks import getRegressMetric, tf_pearson
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

batch_s, name_loss, opt_name, opt_lr = 128, "mse", "adam", 0.00099

years = ["2011"]#, "2013", "2019"]
metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]
delta = [0.5]

C = [1]#np.logspace(6, -6, 13)

alpha = np.logspace(6, -6, 13)

num_splits = 10

features_dir = "features/"

dir_to_save = "outputs/fine_tunned/regressor/"

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
      curr_loss = []
      curr_pearson = []
      curr_mae = []
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
        X, Y = evalClass(X_, Y_, delta_i, type_m="regression")

        print("Delta:", delta_i, "X:", X.shape, "Y:", Y.shape)
        
        best_loss = []
        best_mae = []
        best_mse = []
        best_pearson = []
        
        delta_dir = name_dir+str(int(delta_i*100))
        verifyDir(delta_dir+"/")
        verifyDir(delta_dir+"/logs/")
        verifyDir(delta_dir+"/models/")
        verifyDir(delta_dir+"/charts/")
        verifyDir(delta_dir+"/data/")
          
        fname_max = open(delta_dir + "/"+"splits.csv", "w")
        fname_max.write("split,c,alpha,train,test\n")
        
        for split in range(num_splits):
          X_train, X_val, X_test, Y_train, Y_val, Y_test = getRegressSplit(X, Y)
          
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
              cnn = Dense(1, activation="linear")(model)

              dcnn_model = Model(inputs=model, outputs=cnn, name="famv-regressor")
              for layer in dcnn_model.layers:
                print(layer, layer.trainable)
              dcnn_model.summary()
        
              opt = getOptimizer(opt_name, opt_lr)
              loss = getLoss(dcnn_model, loss_name=name_loss, param_c=c, alpha=alpha_val)
              lr_scheduler = getLearningRate(alpha=alpha_val, decay_name="other")
              dcnn_model.compile(optimizer = opt, loss="mse", metrics=['mean_squared_error', 'mean_absolute_error', tf_pearson])
        
              checkpoint_file = delta_dir+"/models/best_alpha_c.h5"
              checkpoints = ModelCheckpoint(checkpoint_file, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
              
              lrate = LearningRateScheduler(lr_scheduler, verbose=1)
              
              earlyS = EarlyStopping(monitor='mean_squared_error', patience=25, verbose=1, mode='min')
              
              dcnn_history = dcnn_model.fit(X_train, Y_train, batch_size=batch_s, verbose=1, epochs=100, validation_split=0.0, shuffle=True, callbacks = [checkpoints, earlyS, lrate])
              
              print("doing evaluations")
              
              model_to_predict = load_model(checkpoint_file, custom_objects={'loss': loss, 'tf_pearson': tf_pearson})
              
              evaluations = model_to_predict.evaluate(X_val, Y_val, batch_size=batch_s, verbose=1)
              metric_loss = model_to_predict.metrics_names[0]
              loss_value = evaluations[0]
              metric_mse = model_to_predict.metrics_names[0]
              mse_value = evaluations[1]
              metric_mae = model_to_predict.metrics_names[2]
              mae_value = evaluations[2]
              metric_pearson = model_to_predict.metrics_names[3]
              pearson_value = evaluations[3]
              print("%s: %.2f%%" % (metric_loss, loss_value*100))
              print("%s: %.2f%%" % (metric_mse, mse_value*100))
              print("%s: %.2f%%" % (metric_mae, mae_value*100))
              print("%s: %.2f%%" % (metric_pearson, pearson_value*100))
              
              p_train_score = pearson_value
              p_score, _, _ = getRegressMetric(model_to_predict, X_val, Y_val)
              print("%s: %.2f%%" % ("p", p_score*100))
              print("%s: %.2f%%" % ("R^2", p_score**2*100))
              
              fname_split.write(str(c)+","+str(alpha_val) +","+str(round(p_train_score,5)) +"," +str(round(p_score,5)) +"\n")

              scores_val_alpha.append(p_score)
              
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
          
          model = Input(shape=(512,))
          cnn = Dense(1, activation="linear")(model)
          
          dcnn_model = Model(inputs=model, outputs=cnn, name="famv-regressor")
          for layer in dcnn_model.layers:
            print(layer, layer.trainable)
          dcnn_model.summary()
          
          opt = getOptimizer(opt_name, opt_lr)
          loss = getLoss(dcnn_model, loss_name=name_loss, param_c=c, alpha=alpha_val)
          lr_scheduler = getLearningRate(alpha=alpha_val, decay_name="other")
          dcnn_model.compile(optimizer = opt, loss="mse", metrics=['mean_squared_error', 'mean_absolute_error', tf_pearson])
          
          checkpoint_file = delta_dir+"/models/best_"+str(split)+".h5"
          checkpoints = ModelCheckpoint(checkpoint_file, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
              
          lrate = LearningRateScheduler(lr_scheduler, verbose=1)
          
          earlyS = EarlyStopping(monitor='mean_squared_error', patience=25, verbose=1, mode='min')
          
          xtrain_split = shuffle(np.concatenate([X_train, X_val]), random_state=12)
          xtest_split = shuffle(X_test.copy(), random_state=35)

          ytrain_split = shuffle(np.concatenate([Y_train, Y_val]), random_state=12)
          ytest_split = shuffle(Y_test.copy(), random_state=35)
          
          dcnn_history = dcnn_model.fit(xtrain_split, ytrain_split, batch_size=batch_s, verbose=1, epochs=100, validation_split=0.0, shuffle=True, callbacks = [checkpoints, earlyS, lrate])
          
          model_to_predict = load_model(checkpoint_file, custom_objects={'loss': loss, 'tf_pearson': tf_pearson})
              
          evaluations = model_to_predict.evaluate(xtest_split, ytest_split, batch_size=batch_s, verbose=1)
          metric_loss = model_to_predict.metrics_names[0]
          loss_value = evaluations[0]
          metric_mse = model_to_predict.metrics_names[0]
          mse_value = evaluations[1]
          metric_mae = model_to_predict.metrics_names[2]
          mae_value = evaluations[2]
          metric_pearson = model_to_predict.metrics_names[3]
          pearson_value = evaluations[3]
          print("%s: %.2f%%" % (metric_loss, loss_value*100))
          print("%s: %.2f%%" % (metric_mse, mse_value*100))
          print("%s: %.2f%%" % (metric_mae, mae_value*100))
          print("%s: %.2f%%" % (metric_pearson, pearson_value*100))
          
          preds = model_to_predict.predict(xtest_split)
          
          p_train_score = pearson_value
          p_score, _, _ = getRegressMetric(model_to_predict, xtest_split, ytest_split)
          print("%s: %.2f%%" % ("p", p_score*100))
          print("%s: %.2f%%" % ("R^2", p_score**2*100))
          
          clearModel(model_to_predict)
          clearModel(cnn)
          clearModel(model)
          clearModel(dcnn_model)
          clearSession()

          print("Plotting results ...")
          # plot with various axes scales
          plt.figure()
          
          plt.subplot(311)
          plt.plot(dcnn_history.history['mean_absolute_error'],'r')
          #plt.plot(dcnn_history.history['val_mean_absolute_error'],'g')
          plt.yticks(np.arange(0, 1, step=0.05))
          plt.yscale('log')
          plt.xlabel("Epochs")
          plt.ylabel("MAE")
          plt.title("MAE vs Epochs "+data_year+" MAE: "+str(round(mae_value, 3)))
          plt.legend(['Train','Val'])
          plt.yscale('log')
          plt.grid(True)

          plt.subplot(313)
          plt.plot(dcnn_history.history['loss'],'r')
          #plt.plot(dcnn_history.history['val_loss'],'g')
          plt.yticks(np.arange(0, 1, step=0.05))
          plt.yscale('log')
          plt.xlabel("Epochs")
          plt.ylabel("Loss")
          plt.title("Loss vs Epochs "+data_year+" Loss: "+str(round(loss_value, 3)))
          plt.legend(['Train','Val'])
          plt.yscale('log')
          plt.grid(True)
          plt.savefig(name_dir+str(int(delta_i*100))+"_history_mae-loss.png")
          plt.clf()
          plt.cla()
          plt.close()
          
          plt.figure()
          plt.subplot(131)
          plt.plot(ytest_split, ytest_split,'r')
          plt.plot(ytest_split, preds,'bo', alpha=0.1)
          plt.yticks(np.arange(0, 10, step=0.5))
          plt.xticks(np.arange(0, 10, step=0.5))
          plt.title("Regresion y-test pearson="+str(round(p_score,5)))
          plt.legend(["valor real", "valor predicho"])
          plt.grid(True)
          
          best_loss.append(loss_value)
          best_mae.append(mae_value)
          best_mse.append(mse_value)
          best_pearson.append(p_score)
          
          fname_max.write(str(split)+","+str(c_val_max)+","+str(alpha_val_max)+","+ str(round(p_train_score,5)) +"," +str(round(p_score,5)) +"\n")
          
          print("Resume, Train:", p_train_score, "Test:", p_score)
          
        fname_max.close()
        
        curr_loss.append(np.asarray(best_loss).mean())
        curr_mae.append(np.asarray(best_mae).mean())
        curr_mse.append(np.asarray(best_mse).mean())
        curr_pearson.append(np.asarray(best_pearson).mean())
        
        file_log.write("\nDelta: "+ str(delta_i)+"\n")
        file_log.write("X-train: " + str(X_train.shape[0]) + " val: " + str(X_val.shape[0]) + " test: " + str(X_test.shape[0])+"\n")
        file_log.write("Y-train: " + str(Y_train.shape[0]) + " val: " + str(Y_val.shape[0]) + " test: " + str(Y_test.shape[0])+"\n")
        file_log.write(metric_loss+": "+ str(np.asarray(best_loss).mean())+ "\n")
        file_log.write(metric_mae+": "+ str(np.asarray(best_mae).mean())+ "\n")
        file_log.write(metric_mse+": "+ str(np.asarray(best_mse).mean())+ "\n")
        file_log.write("Pearson: "+ str(np.asarray(best_pearson).mean())+ "\n")
        file_log.write("R^2: "+ str(np.asarray(best_pearson).mean()**2)+ "\n")
        
      results[data_year][city][metric] = {'loss': curr_loss,
                                          'mse': curr_mse,
                                          'pearson': curr_pearson,
                                          'mae': curr_mae}
      
      with open(name_dir + "results_summary.json", 'w') as outfile:
        json.dump(results[data_year][city][metric], outfile)
      
      
    for i in range(len(metrics)):
      plt.figure()
      plt.plot(delta, loss[i],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      #plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("Loss")
      plt.title("Testing "+metrics[i]+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metrics[i]+"/fineTuning/loss.png")
      plt.clf()
      plt.cla()
      plt.close()
      
      plt.figure()
      plt.plot(delta, mae[i],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      #plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("MAE")
      plt.title("Testing "+metrics[i]+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metrics[i]+"/fineTuning/mae.png")
      plt.clf()
      plt.cla()
      plt.close()
      
      plt.figure()
      plt.plot(delta, pearson[i],'r')
      #plt.axis([0.0, 0.55, 0.0, 1.0])
      plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, 0.55, step=0.05))
      plt.xlabel("Delta")
      plt.ylabel("Pearson")
      plt.title("Testing "+metrics[i]+" trained in "+city)
      #plt.legend([cities[0], cities[1]])
      plt.grid(True)
      plt.savefig(name_dir_ + city+"/"+metrics[i]+"/fineTuning/pearson.png")
      plt.clf()
      plt.cla()
      plt.close()
      
