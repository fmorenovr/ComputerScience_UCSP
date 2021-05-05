#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc, accuracy_score, f1_score

def getProbs(model, x_test):
  if hasattr(model, "decision_function"):
    Z = model.decision_function(x_test)
  else:
    Z = model.predict_proba(x_test)[:, 1]
  return Z

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
    
  return [metric_value, aux_1, aux_2]

def getClassMetrics(model, x_test, y_test, type_m="PR"):

  [auc, precision, recall] = getCurve(model, x_test, y_test, type_m)
  
  y_pred = model.predict(x_test)
  
  f1 = f1_score(y_test, y_pred)
  acc = accuracy_score(y_test, y_pred)
  
  return [auc, precision, recall], f1, acc

def getClassifier(m, c=1, g=1):
  if m == "SGD":
    svr = SGDClassifier(alpha=c, loss="hinge")
  elif m == "Bayes":
    svr = GaussianNB()
  elif m == "Tree":
    svr = DecisionTreeClassifier()
  elif m == "Forest":
    svr = RandomForestClassifier()
  elif m == "Extra":
    svr = ExtraTreesClassifier()
  elif m == "Ada":
    svr = AdaBoostClassifier()
  elif m == "GBDT":
    svr = GradientBoostingClassifier()
  elif m == "HistGBDT":
    svr = HistGradientBoostingClassifier()
  elif m == "RidgeClassifier":
    svr = RidgeClassifier(alpha=c)
  elif m == "LogisticRegression":
    svr = LogisticRegression(C=c, penalty='l2', solver='liblinear')
  elif m == "Perceptron":
    svr = Perceptron(alpha=c, penalty='l2')
  elif m == "MLP":
    svr = MLPClassifier(alpha=c)
  elif m == "LinearSVC":
    svr = LinearSVC(C=c, loss="hinge")
  elif m == "SVC":
    svr = SVC(kernel='linear', C=c)
  elif m == "NuSVC":
    svr = NuSVC(kernel='linear', nu=0.1)
  elif m=="RBF":
    svr = SVC(kernel="rbf", C=c, gamma=g)
  else:
    svr = LinearSVC(C=c, loss="hinge")
  return svr

