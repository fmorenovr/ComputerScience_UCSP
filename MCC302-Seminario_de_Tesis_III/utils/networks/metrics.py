#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc, accuracy_score, f1_score

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from keras import backend as K
import tensorflow as tf
from keras import losses

def getLoss(model, alpha=1, param_c=1, loss_name="squared_hinge", epsilon=1e-01):
  if loss_name=="squared_hinge":
    def squared_hinge(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      #loss = param_c*K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)
      loss = param_c*losses.squared_hinge(y_true, y_pred)
      
      return regularization + loss
    return squared_hinge

  elif loss_name=="hinge":
    def hinge(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      #regularization = 1/2*alpha*K.sum(K.square(weights))
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      loss = param_c*losses.hinge(y_true, y_pred)
      
      return regularization + loss
    return hinge
  
  elif loss_name=="binary_crossentropy":
    def binary_crossentropy(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      loss = param_c*losses.binary_crossentropy(y_true, y_pred)
    
      return regularization + loss
    return binary_crossentropy
  
  elif loss_name=="epsilon_sensitive":
    def epsilon_sensitive(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      loss = param_c*tf.reduce_sum(K.maximum(K.abs(y_true - y_pred) - epsilon, 0.))
    
      return regularization + loss
    return epsilon_sensitive
  
  elif loss_name=="huber":
    def huber(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      
      error = y_pred_f - y_true_f
      abs_error = K.abs(error)
      quadratic = K.minimum(abs_error, epsilon)
      linear = abs_error - quadratic
      loss = param_c*losses.huber(y_true, y_pred)#(0.5 * K.square(quadratic) + epsilon * linear)
    
      return regularization + loss
    return huber
  
  elif loss_name=="mse":
    def mse(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      
      weights = model.layers[-1].get_weights()[0]
      #bias = model.layers[-1].get_weights()[1]
      
      regularization = 1/2*alpha*tf.reduce_sum(tf.square(weights))
      loss = param_c*losses.mean_squared_error(y_true - y_pred)
    
      return regularization + loss
    return mse

def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  return true_positives / (predicted_positives + K.epsilon())

def f1(y_true, y_pred):
  precision_ = precision(y_true, y_pred)
  recall_ = recall(y_true, y_pred)
  return 2*((precision_*recall_)/(precision_+recall_+K.epsilon()))

def tf_pearson(y_true, y_pred):
  x = y_true
  y = y_pred
  mx = K.mean(x, axis=0)
  my = K.mean(y, axis=0)
  xm, ym = x - mx, y - my
  r_num = K.sum(xm * ym)
  x_square_sum = K.sum(xm * xm)
  y_square_sum = K.sum(ym * ym)
  r_den = K.sqrt(x_square_sum * y_square_sum)
  r = r_num / r_den
  return K.mean(r)

def getProbs(model, x_test):
  return model.predict(x_test)[:, 0]

def getCurve(model, x_test, y_test, type_m="PR"):
  y_scores = getProbs(model, x_test)
  if type_m=="ROC":
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    metric_value = roc_auc_score(y_test, y_scores) #auc(fpr, tpr)
    aux_1, aux_2 = fpr, tpr
  elif type_m=="PR":
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    metric_value = average_precision_score(y_test, y_scores) #auc(recall, precision)
    aux_1, aux_2 = precision, recall
  else:
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    metric_value = roc_auc_score(y_test, y_scores) #auc(fpr, tpr)
    aux_1, aux_2 = fpr, tpr
    
  return metric_value, aux_1, aux_2

def getClassMetrics(model, x_test, y_test, type_m="PR"):

  [auc, precision, recall] = getCurve(model, x_test, y_test, type_m)
  
  y_pred = model.predict(x_test)[:, 0]
  #f1 = f1_score(y_test, y_pred)
  #acc = accuracy_score(y_test, y_pred)
  #print(f1, acc)
  return [auc, precision, recall]

def getRegressMetric(model, x_test, y_test):
  y_scores = getProbs(model, x_test)
  pearson, _ = pearsonr(y_test, y_scores)
  mse = mean_squared_error(y_test, y_scores)
  r2 = r2_score(y_test, y_scores)
  
  return pearson, mse, r2
