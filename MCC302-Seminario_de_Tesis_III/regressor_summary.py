#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import json
import numpy as np
from numpy import genfromtxt

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from joblib import dump, load
from scipy.io import savemat, loadmat

from utils import verifyDir
from utils.datasets import evalClass, getRegressSplit
from utils.libsvm import getRegressMetrics, getRegressor
from utils.preprocessing import getFeatures

years = ["2011", "2013", "2019"]
metrics = ["safety", "wealthy", "uniquely"]
cities = ["Boston", "New York City"]
delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

C = np.logspace(7, -7, 15)

num_splits = 10

features_dir = "features/"

methods_type = "linear"
output_dir = "outputs/summary/regressor_"+methods_type+"/"

features = ['gist', "vgg16_places", 'vgg16_gap_places', 'vgg16', 'vgg16_gap']#, 'fisher']
colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'red', 'blue', 'peru', 'olivedrab', 'darkturquoise', 'fuchsia', 'indigo', 'crimsom', 'darksalmon']

methods = ['Lasso', 'Ridge', 'LinearRegression', 'LinearSVR', 'Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT", "HistGBDT"]#, 'SVR', 'MLP', 'SGD', 'NuSVR']

methods = ['Lasso', 'Ridge', 'LinearRegression', 'LinearSVR'] if methods_type == "linear" else ['Tree', "Forest", "Bayes", "Ada", "Extra", "GBDT"]

stand = ['none']#, 'standard']
reduct = ['none']#, 'PCA']

colors = colors[:len(methods)]

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
              pca = PCA(n_components=0.95, svd_solver='full')#PCA(n_components=n_components)
              X_r = pca.fit_transform(X_s)
              print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
              print("X.shape reducted: ", X.shape)
            else:
              X_r = X_s.copy()              

            for m in methods:
              X_m = X_r.copy()
              Y_m = Y_.copy()
              print("Method:", m)
              print("X:", X_m.shape, "Y:", Y_m.shape)
            
              namefile_m =  s +"_" + r

              root_dir = rof_dir+m+"/"
              verifyDir(root_dir)
          
              f_html = open('{}/results.html'.format(root_dir), 'w');
              f_html.write('<html><body>');
              f_html.write('<h3>{} trained over {}-{} with metric {} and method {}</h3>'.format(f, city, year, metric, m));
              f_html.write('<img src="results.png"/><br/>');
              f_html.write('<table border>');
              
              delta_best_rms = []
              
              for delta_i in delta:
              
                X, y = evalClass(X_m, Y_m, delta_i, type_m="regression")

                print("Delta:", delta_i, "X:", X.shape, "Y:", y.shape)
                best_p_train = []
                best_p_test = []
                best_mse_test = []
                best_r2_test = []
                
                f_html.write('<tr><td><b>delta = {}</b></td>'.format(delta_i));
              
                name_dir = root_dir+str(int(delta_i*100))
                verifyDir(name_dir+"/")
                verifyDir(name_dir+"/data/")
                verifyDir(name_dir+"/logs/")
                verifyDir(name_dir+"/models/")
                verifyDir(name_dir+"/charts/")
          
                fname_max = open(name_dir + "/splits.csv", "w")
                fname_max.write("split,c,train,test\n")
                
                for split in range(num_splits):
                  
                  xtrain, xval, xtest, ytrain, yval, ytest = getRegressSplit(X, y, validation=True)
                  
                  print("Split:", split)
                  print("xtrain:", xtrain.shape, "xval:", xval.shape, "xtest:", xtest.shape)
                  print("ytrain:", ytrain.shape, "yval:", yval.shape, "ytest:", ytest.shape)
                  
                  fname_split = open(name_dir + "/logs/"+namefile_m+"_"+str(split)+".csv", "w")
                  fname_split.write("c,train,val\n")
                  
                  scores_val = []
                  
                  for c in C:
                    print("Evaluating", f, "with method", m, "with c=",c)
                    svr = getRegressor(m, c)
                    svr_model = svr.fit(xtrain, ytrain)
                    
                    # R, mse, mrsq
                    p_train, _, _ = getRegressMetrics(svr_model, xtrain, ytrain)
                    p_val, _, _ = getRegressMetrics(svr_model, xval, yval)

                    scores_val.append(p_val)
                    
                    print("train:", p_train, "val:", p_val)
                    
                    fname_split.write(str(c) + ","+str(round(p_train,5)) +"," +str(round(p_val,5)) +"\n")
                    
                  fname_split.close()

                  scores_val_ = np.asarray(scores_val)
                  nan_array = np.isnan(scores_val_)
                  not_nan_array = ~ nan_array
                  scores_val_ = scores_val_[not_nan_array]
                  C_ = C[not_nan_array]
                  
                  index_max_test = np.argmax(np.asarray(scores_val_))
                  c_val = C_[index_max_test]
                  
                  svr = getRegressor(m, c_val)

                  xtrain_split = shuffle(np.concatenate([xtrain, xval]), random_state=12)
                  xtest_split = shuffle(xtest.copy(), random_state=35)

                  ytrain_split = shuffle(np.concatenate([ytrain, yval]), random_state=12)
                  ytest_split = shuffle(ytest.copy(), random_state=35)

                  print("xtrain:", xtrain.shape, "xtest:", xtest.shape)
                  print("ytrain:", ytrain.shape, "ytest:", ytest.shape)

                  svr_model = svr.fit(xtrain_split, ytrain_split)

                  p_train, _, _ = getRegressMetrics(svr_model, xtrain_split, ytrain_split)
                  
                  p_test, mse_test, r2_test = getRegressMetrics(svr_model, xtest_split, ytest_split)
                  
                  ypred_split = svr_model.predict(xtest_split)
                  
                  plt.figure()
                  plt.plot(ytest_split, ytest_split,'r')
                  plt.plot(ytest_split, ypred_split,'bo', alpha=0.1)
                  plt.yticks(np.arange(0, 10, step=0.5))
                  plt.xticks(np.arange(0, 10, step=0.5))
                  plt.title("Regresion y-test pearson="+str(round(p_test,5)))
                  plt.legend(["valor real", "valor predicho"])
                  plt.grid(True)
                  plt.savefig(name_dir + "/charts/"+namefile_m+"_"+str(split)+".png")
                  plt.clf()
                  plt.cla()
                  plt.close()
                  print("Resume: Train:", p_train, "Test:", p_test)
                  
                  best_p_train.append(round(p_train,5))
                  best_p_test.append(round(p_test,5))
                  best_mse_test.append(round(mse_test,5))
                  best_r2_test.append(round(r2_test,5))
                  
                  dump(svr_model, name_dir + "/models/"+namefile_m+"_"+str(c_val)+"_"+str(split)+'.joblib')
                  
                  img_delta_path = str(int(delta_i*100)) + "/charts/"+namefile_m+"_"+str(split)+".png"
                  f_html.write('<td><a>');
                  f_html.write('<img src="{}" height="120" width="120"/></a><br/>P={}, MSE={}, R^2={}</td>'.format(img_delta_path, round(p_test,5), round(mse_test,5), round(r2_test,5)));
                  
                  fname_max.write(str(split)+","+str(c_val)+","+ str(round(p_train,5)) +"," +str(round(p_test,5)) +"\n")
                  
                fname_max.close()
                
                best_p_train = np.asarray(best_p_train)
                best_p_test = np.asarray(best_p_test)
                best_mse_test = np.asarray(best_mse_test)
                best_r2_test = np.asarray(best_r2_test)
                
                print("Mean Train:", best_p_train.mean())
                print("Mean Test:", best_p_test.mean())
                
                rms = np.sqrt(np.mean(np.power(best_p_test, 2)))
                
                delta_best_rms.append(round(rms,5))
              
                f_html.write('<td><br><span style="color:blue">mean(P)</span>={} ({})</br><br><span style="color:red">mean(MSE)</span>={} ({})</br><br><span style="color:green">mean(R^2)</span>={} ({})</br></td>'.format(round(best_p_test.mean(),5), round(best_p_test.std(),5), round(best_mse_test.mean(),5), round(best_mse_test.std(),5), round(best_r2_test.mean(),5), round(best_r2_test.std(),5)));
#                f_html.write('<span style="color:red">mean(AUC)</span> = {} ({})</td>', mean(delta_aucs_harder(delta_ind, :)), std(delta_aucs_harder(delta_ind, :)));
                f_html.write('</tr>');
              
              plt.figure()
              plt.plot(delta, delta_best_rms,'b')
              plt.yticks(np.arange(0, 1.4, step=0.2))
              plt.xticks(np.arange(0, 0.55, step=0.05))
              plt.xlabel("Delta")
              plt.ylabel("Pearson")
              plt.title("Pearson - Delta, trained in {}-{}-{}".format(metric, city, year))
              plt.grid(True)
              plt.savefig(root_dir + "results.png")
              plt.clf()
              plt.cla()
              plt.close()
              
              data_results[f][m] = delta_best_rms.copy()
              
              f_html.write('</table>');
              f_html.write('</body></html>');
              f_html.close();
              
            with open(rof_dir+"results_summary.json", 'w') as outfile:
              json.dump(data_results[f], outfile)
      
      for f in features:
        rof_dir = output_dir+str(year)+"/"+city+"/"+metric+"/"+f+"/"
        plt.figure()
        for m, c in zip(methods, colors):
          print(f, m, c)
          plt.plot(delta, data_results[f][m], label=m, color=c)
        plt.yticks(np.arange(0.6, 1.2, step=0.1))
        plt.xticks(np.arange(0, 0.55, step=0.05))
        plt.xlabel("Delta")
        plt.ylabel("Pearson")
        plt.title("Pearson - Delta, trained in {}-{}-{}".format(metric, city, year))
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(rof_dir+"results_summary.png")
        plt.clf()
        plt.cla()
        plt.close()
        
        f_summary = open('{}/results_summary.html'.format(rof_dir), 'w');
        f_summary.write('<html><body>');
        f_summary.write('<h2>Summary</h2>');
        f_summary.write('<h3>{} trained over {}-{} with metric {}</h3>'.format(f, city, year, metric));
        f_summary.write('<img src="results_summary.png"/><br/>');
        f_summary.write('<table border>');
        f_summary.write('<tr>');
        f_summary.write('<td><b>Metodo\Delta</b></td>');
        for delta_i in delta:
          f_summary.write('<td><b>delta = {}</b></td>'.format(delta_i));
        f_summary.write('</tr>');
        for m in methods:
          f_summary.write('<tr>');
          f_summary.write('<td><a href="{}/results.html"><b>{}</b></a></td>'.format(m, m));
          for num in data_results[f][m]:
            f_summary.write('<td>{}</td>'.format(num));
          f_summary.write('</tr>');
        f_summary.write('</table>');
        f_summary.write('</body></html>');
        f_summary.close();
      
      with open(output_dir+str(year)+"/"+city+"/"+metric+"/results_year.json", 'w') as outfile:
        json.dump(data_results, outfile)
      
      f_summary_year = open('{}/results_year.html'.format(output_dir+str(year)+"/"+city+"/"+metric+"/"), 'w');
      f_summary_year.write('<html><body>');
      f_summary_year.write('<h2>Summary per Year</h2>');
      f_summary_year.write('<h3>Features trained over {}-{} with metric {}</h3>'.format(city, year, metric));
      f_summary_year.write('<table border>');
      f_summary_year.write('<tr>');
      for f in features:
        f_summary_year.write('<td><b>{}, features = {}</b></td>'.format(f, data_results[f]['length']));
      f_summary_year.write('</tr>');
      f_summary_year.write('<tr>');
      for f in features:
        f_summary_year.write('<td><a href="{}/results_summary.html"><img src="{}/results_summary.png"/></a></td>'.format(f, f));
      f_summary_year.write('</tr>');
      f_summary_year.write('</table>');
      f_summary_year.write('</body></html>');
      f_summary_year.close();
#with open('data.json', 'r') as fp:
#  data = json.load(fp)
