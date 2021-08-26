#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from joblib import dump, load
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import permutation_importance
import seaborn as sns
from utils import verifyDir

output_dir_ = "to_test/"

output_dir_pp2 = "to_test_pp2/"

delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]

top_elements = 10

for delta_i in delta:

  output_dir = output_dir_ + str(int(delta_i*100)) + "/"

  data_psp = load(output_dir + "dataset_psp.joblib")
  rbf_psp = load(output_dir +'RBF_psp.joblib')
  linear_psp = load(output_dir +'LinearSVC_psp.joblib')
  log_psp = load(output_dir+'LogisticRegression_psp.joblib')
  #print("The best parameters are %s with a score of %0.2f" % (data.best_params_, data.best_score_))

  data_psp_ = load(output_dir_pp2 + "dataset_psp.joblib")
  rbf_psp_ = load(output_dir_pp2 +'RBF_psp.joblib')
  linear_psp_ = load(output_dir_pp2 +'LinearSVC_psp.joblib')
  log_psp_ = load(output_dir_pp2+'LogisticRegression_psp.joblib')

  
  models = {'rbf_psp': rbf_psp, 'svc_psp':linear_psp, 'log_psp':log_psp}
  preds = {}
  
  models_pp2 = {'rbf_psp': rbf_psp_pp2, 'svc_psp':linear_psp_pp2, 'log_psp':log_psp_pp2}
  preds_pp2 = {}

  xtrain_psp, xtest_psp, ytrain_psp, ytest_psp = data_psp['xtrain_val'], data_psp['xtest'], data_psp['ytrain_val'], data_psp['ytest']

  xtrain_psp_pp2, xtest_psp_pp2, ytrain_psp_pp2, ytest_psp_pp2 = data_psp_pp2['xtrain_val'], data_psp_pp2['xtest'], data_psp_pp2['ytrain_val'], data_psp_pp2['ytest']
  
  model_name = "log"

  preds[model_name+"_psp"] = models[model_name+'_psp'].predict(xtest_psp.iloc[:,1:].to_numpy(copy=True))

  preds_pp2[model_name+"_psp"] = models_pp2[model_name+'_psp'].predict(xtest_psp_pp2.iloc[:,1:].to_numpy(copy=True))

  y_eq_psp = ytest_psp["class"].values == preds[model_name+"_psp"]
  y_eq_psp_pp2 = ytest_psp_pp2["class"].values == preds_pp2[model_name+"_psp"]

  y_eq_pred = [a and b for a, b in zip(y_eq_gap, y_eq_psp)]

  d_class={
           "ID":xtest_psp["ID"].values,
           "y_scores": ytest_psp["safety"].values,
           "y_test": ytest_psp["class"].values,
           "y_pred_psp": preds[model_name+"_psp"],
           "y_eq_psp": y_eq_psp,
           }

  if model_name=='log':
    y_probs_psp = models[model_name+'_psp'].predict_proba(xtest_psp.iloc[:,1:].to_numpy(copy=True))

    y_probs_psp = [b for a,b in y_probs_psp]
    
    d_class["y_prob_psp"] = y_probs_psp

  data_summary = pd.DataFrame(data=d_class)
  #print(data_summary)
  #print(data_summary.sum())
  # filtering per assertion

  confusion_matrix_psp = pd.crosstab(data_summary["y_test"], data_summary["y_pred_psp"], rownames=["test"], colnames=["pred"])

  fig = plt.figure(figsize=(16, 8))
  #sns.set_theme(style="whitegrid")
#  plt.subplot(1, 3, 1)
#  plt.title("Confusion Matrix GAP")
#  sns.heatmap(confusion_matrix_gap, annot=True, fmt="d")

  plt.subplot(1, 3, 2)
  plt.title("Confusion Matrix PSP")
  sns.heatmap(confusion_matrix_psp, annot=True, fmt="d")

#  plt.subplot(1, 3, 3)
#  plt.title("Confusion Matrix GAP Places")
#  sns.heatmap(confusion_matrix_gap_places, annot=True, fmt="d")

  plt.savefig(output_dir+"confusion_matrix_"+model_name+".png", bbox_inches='tight', pad_inches = 0)
  plt.clf()
  plt.cla()
  plt.close()

  correct_preds_df = data_summary[data_summary["y_eq_pred"]==True]
  #print(correct_preds_df)
  #print(xtest_psp)

  data_pred_correct = pd.merge(correct_preds_df, xtest_psp, how="inner", on="ID")
  data_pred_correct.sort_values("y_scores", ascending=False, inplace=True)
  data_pred_correct.to_csv(output_dir+'explanations/data_pred_correct_'+model_name+'.csv', index=False)

  if model_name=='log':
    importance_label = data_pred_correct.columns[12:]
  else:
    importance_label = data_pred_correct.columns[8:]

  importance_value = models[model_name+'_psp'].coef_[0]

  feature_importance = abs(100.0 * (importance_value / importance_value.max()))
  sorted_idx_fi = np.argsort(feature_importance)[::-1]

  # summarize feature importance
  for i,v in enumerate(importance_value):
	  print('Feature: %s, Score: %.5f' % (importance_label[i],v))
  # plot feature importance

  fig = plt.figure(figsize=(16, 6))
  #sns.set_theme(style="whitegrid")
  plt.subplot(1, 4, 1)
  plt.title('Feature Importance')
  plt.barh(importance_label[sorted_idx_fi][:top_elements], importance_value[sorted_idx_fi][:top_elements])
  plt.ylabel('Object')
  plt.xlabel('Value')
  
  if model_name=='log':
    xtest_data_correct = data_pred_correct.iloc[:,12:].copy()
    xtest_data_correct_safe = data_pred_correct[data_pred_correct["y_test"]==1].iloc[:,12:].copy()
    xtest_data_correct_notsafe = data_pred_correct[data_pred_correct["y_test"]==0].iloc[:,12:].copy()
  else:
    xtest_data_correct = data_pred_correct.iloc[:,8:].copy()
    xtest_data_correct_safe = data_pred_correct[data_pred_correct["y_test"]==1].iloc[:,8:].copy()
    xtest_data_correct_notsafe = data_pred_correct[data_pred_correct["y_test"]==0].iloc[:,8:].copy()

  ytest_data_correct = data_pred_correct["y_test"].values
  ytest_data_correct_safe = data_pred_correct[data_pred_correct["y_test"]==1]["y_test"].values
  ytest_data_correct_notsafe = data_pred_correct[data_pred_correct["y_test"]==0]["y_test"].values
  
  xtest_data_correct_ = xtest_data_correct.copy()
  xtest_data_correct_["y_test"] = ytest_data_correct
  xtest_data_correct_.to_csv(output_dir+'explanations/presence.csv', index=False)
  
  xtest_data_correct_safe_ = xtest_data_correct_safe.copy()
  xtest_data_correct_safe_["y_test"] = ytest_data_correct_safe
  xtest_data_correct_safe_.to_csv(output_dir+'explanations/presence_safe.csv', index=False)
  
  xtest_data_correct_notsafe_ = xtest_data_correct_notsafe.copy()
  xtest_data_correct_notsafe_["y_test"] = ytest_data_correct_notsafe
  xtest_data_correct_notsafe_.to_csv(output_dir+'explanations/presence_notsafe.csv', index=False)
  
  plt.subplot(1, 4, 2)

  result = permutation_importance(models[model_name+'_psp'], xtest_data_correct.to_numpy(copy=True), ytest_data_correct, n_repeats=10, random_state=42, n_jobs=2)
  '''
  import eli5
  from eli5.sklearn import PermutationImportance
  
  perm = PermutationImportance(models[model_name+'_psp'], random_state=1).fit(xtest_data_correct.to_numpy(copy=True), ytest_data_correct)
  print(perm.feature_importances_)
  print(result.importances_mean)
  eli5.show_weights(perm, feature_names = importance_label.to_list())
  '''
  
  sorted_idx_pi = result.importances_mean.argsort()[::-1]

  for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
      print("Feature: {}, Mean: {:.3f}, Std: +/- {:.3f}".format(importance_label[i], result.importances_mean[i], result.importances_std[i]))

  plt.boxplot(result.importances[sorted_idx_pi][:top_elements].T, vert=False, labels=importance_label[sorted_idx_pi][:top_elements], notch=True)
  plt.title("Permutation Importance")
  plt.ylabel('Object')
  plt.xlabel('Value')
  fig.tight_layout()

  plt.subplot(1, 4, 3)

  result = permutation_importance(models[model_name+'_psp'], xtest_data_correct_safe.to_numpy(copy=True), ytest_data_correct_safe, n_repeats=10, random_state=42, n_jobs=2)
  
  sorted_idx_pi = result.importances_mean.argsort()[::-1]

  for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
      print("Feature: {}, Mean: {:.3f}, Std: +/- {:.3f}".format(importance_label[i], result.importances_mean[i], result.importances_std[i]))

  plt.boxplot(result.importances[sorted_idx_pi][:top_elements].T, vert=False, labels=importance_label[sorted_idx_pi][:top_elements], notch=True)
  plt.title("Permutation Importance on safe")
  plt.ylabel('Object')
  plt.xlabel('Value')
  fig.tight_layout()

  plt.subplot(1, 4, 4)
  
  result = permutation_importance(models[model_name+'_psp'], xtest_data_correct_notsafe.to_numpy(copy=True), ytest_data_correct_notsafe, n_repeats=10, random_state=42, n_jobs=2)
  
  sorted_idx_pi = result.importances_mean.argsort()[::-1]

  for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
      print("Feature: {}, Mean: {:.3f}, Std: +/- {:.3f}".format(importance_label[i], result.importances_mean[i], result.importances_std[i]))
  
  plt.boxplot(result.importances[sorted_idx_pi][:top_elements].T, vert=False, labels=importance_label[sorted_idx_pi][:top_elements], notch=True)
  plt.title("Permutation Importance on notsafe")
  plt.ylabel('Object')
  plt.xlabel('Value')
  fig.tight_layout()
  

  plt.savefig(output_dir+"object_influence_"+model_name+".png", bbox_inches='tight', pad_inches = 0)
  plt.clf()
  plt.cla()
  plt.close()
  
  ## GENERATING HTML
  components = data_pred_correct.iloc[:,8:].sum()

  sorted_com = sorted(enumerate(components.values), key=lambda x: x[1])

  top_10 = sorted_com[-10:]
  #print(top_10)
  #print(xtest_psp["ID"].values == data_summary["ID"].values)

  root_dir = "data/images/pp1/2011/"

  f_summary = open(('{}/results_summary_'+model_name+'.html').format(output_dir), 'w');
  f_summary.write('<html><body>');
  f_summary.write('<h2>Summary</h2>');
  f_summary.write('<h3>Images correctly classifies, trained over {}-{} with metric {}</h3>'.format("Boston", "2011", "Safety"));

  f_summary.write('<table border>');

  f_summary.write('<tr>');
  for i in range(9,-1,-1):
    f_summary.write('<td><b>{}</b></td>'.format(components.index[top_10[i][0]]));
  f_summary.write('</tr>');

  f_summary.write('<tr>');
  for i in range(9,-1,-1):
    f_summary.write('<td><b>{}</b></td>'.format(top_10[i][1]));
  f_summary.write('</tr>');

  f_summary.write('</table>');
  f_summary.write('<table border>');
  f_summary.write('<tr>');
  f_summary.write('<td><b>Img ID</b></td>');
  f_summary.write('<td><b>Class</b></td>');
  f_summary.write('<td><b>Score</b></td>');
  if model_name=='log':
    f_summary.write('<td><b>Prob PSP</b></td>');
  for i in range(10):
    f_summary.write('<td><b>Component {}</b></td>'.format(i+1));
  f_summary.write('</tr>');

  #for i in range(10):
  #  f_summary.write('<td><b>{}</b></td>'.format(delta_i));
  #f_summary.write('</tr>');
  for img in data_pred_correct["ID"].values:
    f_summary.write('<tr>');
    f_summary.write('<td><img src="./../../{}"/></td>'.format(root_dir+str(img)+".jpg"));
    row_img = data_pred_correct[data_pred_correct["ID"]==img]
    #print(row_img.values[0][8:])
    if model_name=='log':
      componentes_img = row_img.values[0][12:]
      columns_name = row_img.columns[12:]
    else:
      componentes_img = row_img.values[0][8:]
      columns_name = row_img.columns[8:]
    sorted_com_img = sorted(enumerate(componentes_img), key=lambda x: x[1])
    top_10_img = sorted_com_img[-10:]
    f_summary.write('<td>{}</td>'.format(row_img["y_test"].values[0]));
    f_summary.write('<td>{}</td>'.format(row_img["y_scores"].values[0]));
    if model_name=='log':
      f_summary.write('<td>{}</td>'.format(row_img["y_prob_psp"].values[0]));
    for i in range(9,-1,-1):
      f_summary.write('<td><b>{} ({})</b></td>'.format(top_10_img[i][1], columns_name[top_10_img[i][0]]));
  #  for num in data_results[f][m]:
  #    f_summary.write('<td>{}</td>'.format(num));
    f_summary.write('</tr>');

  f_summary.write('</body></html>');
  f_summary.close();

  '''
  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)


  width = 4000
  height = 3000
  max_dim = 100

  full_image = Image.new('RGBA', (width, height))
  '''

  ## charts

  print("Drawing objects")

  fig = plt.figure(figsize=(16, 6))
  #sns.set_theme(style="whitegrid")
  plt.subplot(1, 4, 1)
  plt.title('Object presence')

  object_presence = (xtest_data_correct>0).sum().values
  sorted_idx_op = np.argsort(object_presence)[::-1]

  plt.barh((importance_label[sorted_idx_op])[:top_elements], (object_presence[sorted_idx_op])[:top_elements])
  plt.xlabel('# images where appear')
  plt.ylabel('Object')
  
  plt.subplot(1, 4, 2)
  
  object_presence_m = xtest_data_correct.mean().values
  sorted_idx_op_m = np.argsort(object_presence_m)[::-1]
  
  #sns.set_theme(style="ticks", color_codes=True)
  plt.title("Object Presence")
  plt.boxplot((xtest_data_correct.to_numpy(copy=True)[:,sorted_idx_op_m])[:,:top_elements], vert=False, labels=(importance_label[sorted_idx_op_m])[:top_elements], notch=True)
  plt.xlabel('% of presences in images where appear')
  plt.ylabel('Object')
  fig.tight_layout()

  sorted_idx_msaf = np.argsort(xtest_data_correct_safe.mean().values)[::-1]
  
  plt.subplot(1, 4, 3)
  #sns.set_theme(style="ticks", color_codes=True)
  plt.title("Object Presence in safe")
  plt.boxplot((xtest_data_correct_safe.to_numpy(copy=True)[:,sorted_idx_msaf])[:,:top_elements], vert=False, labels=(importance_label[sorted_idx_msaf])[:top_elements], notch=True)
  plt.xlabel('% of presences in images where appear')
  plt.ylabel('Object')
  fig.tight_layout()

  sorted_idx_mnsaf = np.argsort(xtest_data_correct_notsafe.mean().values)[::-1]

  plt.subplot(1, 4, 4)
  plt.title("Object Presence in not safe")
  plt.boxplot((xtest_data_correct_notsafe.to_numpy(copy=True)[:,sorted_idx_mnsaf])[:,:top_elements], vert=False, labels=(importance_label[sorted_idx_mnsaf])[:top_elements], notch=True)
  plt.xlabel('% of presences in images where appear')
  plt.ylabel('Object')
  fig.tight_layout()

  plt.savefig(output_dir+"object_presence.png", bbox_inches='tight', pad_inches = 0)
  plt.clf()
  plt.cla()
  plt.close()

  # explanationz

  print("LIME Explaining")

  try:
      import lime
  except:
      import sys
      sys.path.append(os.path.join('..', '..')) # add the current directory
      import lime
  from lime import lime_tabular

  explainer  = lime_tabular.LimeTabularExplainer(xtrain_psp.iloc[:,1:].to_numpy(copy=True), mode='classification', training_labels=ytrain_psp["class"].values, class_names =["unsafe", "safe"], feature_names=importance_label, discretize_continuous=False, verbose=True)

  if model_name=='log':
    probs = models[model_name+'_psp'].predict_proba
  else:
    probs = models[model_name+'_psp'].predict_proba

  xtest_safe = xtest_data_correct_safe.to_numpy(copy=True)

  importance_LIME_safe = pd.DataFrame(data={"features": xtest_data_correct_safe.columns})
  
  verifyDir(output_dir+"explanations/")
  
  for i in range(len(xtest_data_correct_safe)):
  
    exp = explainer.explain_instance(xtest_safe[i], probs, num_features=80, top_labels=3, num_samples=5000)

    aux = pd.DataFrame(exp.as_list()).transpose()
    
    sorted_idx_limesafe = np.argsort(aux.iloc[1,:].values)[::-1]
    aux_class = {"features": aux.iloc[0,:].values[sorted_idx_limesafe], str(i): aux.iloc[1,:].values[sorted_idx_limesafe]}
    
    aux_df = pd.DataFrame(data=aux_class)
    
    importance_LIME_safe = pd.merge(importance_LIME_safe, aux_df, how="inner", on="features")

    #exp.save_to_file(output_dir+'explanations/explanation_safe_'+str(i)+'.html')

    exp.as_pyplot_figure()
    from matplotlib import pyplot as plt
    plt.tight_layout()
    #plt.savefig(output_dir+"explanations/explanation_safe_"+str(i)+".png", bbox_inches='tight', pad_inches = 0)
    plt.clf()
    plt.cla()
    plt.close()
    
  importance_LIME_safe.to_csv(output_dir+'explanations/importance_safe.csv', index=False)
  
  xtest_notsafe = xtest_data_correct_notsafe.to_numpy(copy=True)
  
  importance_LIME_notsafe = pd.DataFrame(data={"features": xtest_data_correct_notsafe.columns})
  
  for i in range(len(xtest_data_correct_notsafe)):
  
    exp = explainer.explain_instance(xtest_notsafe[i], probs, num_features=80, top_labels=3, num_samples=5000)
    
    aux = pd.DataFrame(exp.as_list()).transpose()
    sorted_idx_limenotsafe = np.argsort(aux.iloc[1,:].values)[::-1]
    aux_class = {"features": aux.iloc[0,:].values[sorted_idx_limenotsafe], str(i): aux.iloc[1,:].values[sorted_idx_limenotsafe]}

    aux_df = pd.DataFrame(data=aux_class)
    
    importance_LIME_notsafe = pd.merge(importance_LIME_notsafe, aux_df, how="inner", on="features")

    #exp.save_to_file(output_dir+'explanations/explanation_notsafe_'+str(i)+'.html')

    exp.as_pyplot_figure()
    from matplotlib import pyplot as plt
    plt.tight_layout()
    #plt.savefig(output_dir+"explanations/explanation_notsafe_"+str(i)+".png", bbox_inches='tight', pad_inches = 0)
    plt.clf()
    plt.cla()
    plt.close()
    
  importance_LIME_notsafe.to_csv(output_dir+'explanations/importance_notsafe.csv', index=False)

  '''
  # Code for SP-LIME
  import warnings
  from lime import submodular_pick

  # Remember to convert the dataframe to matrix values
  # SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
  sp_obj = submodular_pick.SubmodularPick(explainer, df_titanic[model.feature_name()].values, \
  prob, num_features=5,num_exps_desired=10)

  [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]
  '''

  data_safe_raw = pd.read_csv(output_dir+'explanations/importance_safe.csv')#.iloc[:10,:]
  data_safe = data_safe_raw.copy()
  rows, columns = data_safe.shape
  data_safe["class"] = ["safe"] * rows
  data_safe = pd.melt(data_safe, id_vars=["features", "class"])

  data_notsafe_raw = pd.read_csv(output_dir+'explanations/importance_notsafe.csv')#.iloc[:10,:]
  data_notsafe = data_notsafe_raw.copy()
  rows, columns = data_notsafe.shape
  data_notsafe["class"] = ["notsafe"] * rows
  data_notsafe = pd.melt(data_notsafe, id_vars=["features", "class"])

  fig = plt.figure(figsize=(12, 6))
  #sns.set_theme(style="whitegrid")
  plt.subplot(1, 4, 1)
  plt.title("Object Coefficients")
  
  points = models[model_name+'_psp'].coef_[0]
  sorted_idx_coeff = np.argsort(abs(points))[::-1]
  #df_coeff = pd.DataFrame(data={"labels":importance_label,"value":points})
  ##sns.set_theme(style="ticks", color_codes=True)
  #sns.catplot(x="value", y="labels", data=df_coeff)
  plt.boxplot(points[sorted_idx_coeff][:top_elements].reshape(-1,1).T, vert=False, labels=importance_label[sorted_idx_coeff][:top_elements], notch=True)
  plt.xlabel('level of importances in model')
  plt.ylabel('Object')
  fig.tight_layout()
  
  plt.subplot(1, 4, 2)
  plt.title("Object Importance")
  
  df_imp = data_safe_raw.set_index('features').transpose().copy().append(data_notsafe_raw.set_index('features').transpose().copy(), ignore_index=True)

  df_imp.to_csv(output_dir+'explanations/importance.csv', index=False)
  
  sorted_idx_med = np.argsort(abs(df_imp.median()).values)[::-1]
#  #sns.set_theme(style="ticks", color_codes=True)

#  sns.catplot(x="value", y="features", data=df, hue="class")
  plt.boxplot((df_imp.to_numpy(copy=True)[:,sorted_idx_med])[:,:top_elements], vert=False, labels=importance_label[sorted_idx_med][:top_elements], notch=True)
  plt.xlabel('level of importances in predicted images')
  plt.ylabel('Object')
  fig.tight_layout()

  df_safe = data_safe_raw.set_index('features').transpose().copy()
  #importance_label = df_safe.columns

  sorted_idx_med_safe = np.argsort(abs(df_safe.median()).values)[::-1]

  plt.subplot(1, 4, 3)
  plt.title("Object Importance in safe")
  plt.boxplot((df_safe.to_numpy(copy=True)[:,sorted_idx_med_safe])[:,:top_elements], vert=False, labels=importance_label[sorted_idx_med_safe][:top_elements], notch=True)
  plt.xlabel('level of importances in predicted images')
  plt.ylabel('Object')
  fig.tight_layout()

  df_notsafe = data_notsafe_raw.set_index('features').transpose().copy()

  sorted_idx_med_notsafe = np.argsort(abs(df_notsafe.median()).values)[::-1]

  plt.subplot(1, 4, 4)
  plt.title("Object Importance in not safe")
  plt.boxplot((df_notsafe.to_numpy(copy=True)[:,sorted_idx_med_notsafe])[:,:top_elements], vert=False, labels=importance_label[sorted_idx_med_notsafe][:top_elements], notch=True)
  plt.xlabel('level of importances in predicted images')
  plt.ylabel('Object')
  fig.tight_layout()

  plt.savefig(output_dir+"object_importance_"+model_name+".png", bbox_inches='tight', pad_inches = 0)
  plt.clf()
  plt.cla()
  plt.close()

  plt.figure()
  #sns.set_theme(style="whitegrid")

  df_transpose = data_safe_raw.set_index('features').transpose().copy()
  rows, columns = df_transpose.shape
  columns = df_transpose.columns
  df_transpose["features"] = range(rows)

  # Make the PairGrid
  g = sns.PairGrid(df_transpose, x_vars=columns, y_vars=["features"], aspect=.25, height=10)

  # Draw a dot plot using the stripplot function
  g.map(sns.stripplot, size=6, orient="h", palette="Dark2_r", linewidth=1, edgecolor="w")

  # Use the same x axis limits on all columns and add better labels
  #g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

  titles = columns

  for ax, title in zip(g.axes.flat, titles):
      ax.set(title=title)
      ax.xaxis.grid(False)
      ax.yaxis.grid(True)

  sns.despine(left=True, bottom=True)
  plt.savefig(output_dir+"object_importance_safe_"+model_name+".png")
  plt.clf()
  plt.cla()
  plt.close()

  plt.figure()
  #sns.set_theme(style="whitegrid")

  df_transpose = data_notsafe_raw.set_index('features').transpose().copy()
  rows, columns = df_transpose.shape
  columns = df_transpose.columns
  df_transpose["features"] = range(rows)

  # Make the PairGrid
  g = sns.PairGrid(df_transpose, x_vars=columns, y_vars=["features"], aspect=.25, height=10)

  # Draw a dot plot using the stripplot function
  g.map(sns.stripplot, size=6, orient="h", palette="Dark2_r", linewidth=1, edgecolor="w")

  # Use the same x axis limits on all columns and add better labels
  #g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

  titles = columns

  for ax, title in zip(g.axes.flat, titles):
      ax.set(title=title)
      ax.xaxis.grid(False)
      ax.yaxis.grid(True)

  sns.despine(left=True, bottom=True)
  plt.savefig(output_dir+"object_importance_notsafe_"+model_name+".png")
  plt.clf()
  plt.cla()
  plt.close()
  
