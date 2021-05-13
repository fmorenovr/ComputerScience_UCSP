#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

#from xgboost import XGBRegressor

from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.

def getRegressMetrics(model, x_test, y_test):
  y_pred = model.predict(x_test)
  pearson, _ = pearsonr(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  
  return pearson, mse, r2

def getRegressor(m, c):
  if m == "SGD": #l2
    svr = SGDRegressor(alpha=c, loss="epsilon_insensitive")
  #elif m == "xgboost":
    #svr = = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
  elif m == "Bayes":
    svr = BayesianRidge()
  elif m == "Tree":
    svr = DecisionTreeRegressor()
  elif m == "Forest":
    svr = RandomForestRegressor()
  elif m == "Extra":
    svr = ExtraTreesRegressor()
  elif m == "Ada":
    svr = AdaBoostRegressor()
  elif m == "GBDT":
    svr = GradientBoostingRegressor()
  elif m == "HistGBDT":
    svr = HistGradientBoostingRegressor()
  elif m == "Ridge":
    svr = Ridge(alpha=c)
  elif m == "Lasso":
    svr = Lasso(alpha=c) #l1
  elif m == "LinearRegression":
    svr = LinearRegression() #l0
  elif m == "MLP":
    svr = MLPRegressor(alpha=c) #l2
  elif m == "LinearSVR":
    svr = LinearSVR(C=c, loss='squared_epsilon_insensitive', dual=False)
  elif m == "SVR":
    svr = SVR(kernel='linear', C=c)
  elif m == "NuSVR":
    svr = NuSVR(kernel='linear', C=c, nu=0.1)
  else:
    svr = LinearSVR(C=c, loss='squared_epsilon_insensitive', dual=False)
  return svr
