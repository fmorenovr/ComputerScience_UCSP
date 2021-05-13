import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump, load

delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

output_dir = input_dir = 'to_test/'

data_to_save = {}

top_elements = 10

df_safe_global = pd.DataFrame()
df_notsafe_global = pd.DataFrame()
df_global = pd.DataFrame()

df_presence_safe = pd.DataFrame()
df_presence_notsafe = pd.DataFrame()
df_presence_global = pd.DataFrame()

for delta_i in delta:
  str_name = str(int(delta_i*100))
  
  ## importance
  
  data_safe = pd.read_csv(input_dir+str_name+'/explanations/importance_safe.csv')
  data_safe = data_safe.set_index('features').transpose().copy()
  data_safe_median = data_safe.mean()
  
  data_notsafe = pd.read_csv(input_dir+str_name+'/explanations/importance_notsafe.csv')
  data_notsafe = data_notsafe.set_index('features').transpose().copy()
  data_notsafe_median = data_notsafe.mean()

  labels = data_safe.columns.to_list()
  aux_df = pd.DataFrame()
  aux_df = aux_df.append(data_safe, ignore_index=True)
  aux_df = aux_df.append(data_notsafe, ignore_index=True)
  
  df_safe_global[str_name] = data_safe_median
  df_notsafe_global[str_name] = data_notsafe_median
  df_global[str_name] = aux_df.mean()
  
  ## presence
  
  df_safe_p = pd.read_csv(input_dir+str_name+'/explanations/presence_safe.csv')
  df_safe_y = df_safe_p.iloc[:,-1]
  df_safe_p = df_safe_p.iloc[:,:-1].copy()
  data_p_safe_median = df_safe_p.mean()

  df_notsafe_p = pd.read_csv(input_dir+str_name+'/explanations/presence_notsafe.csv')
  df_notsafe_y = df_notsafe_p.iloc[:,-1]
  df_notsafe_p = df_notsafe_p.iloc[:,:-1].copy()
  data_p_notsafe_median = df_notsafe_p.mean()
  
  aux_df = pd.DataFrame()
  aux_df = aux_df.append(df_safe_p, ignore_index=True)
  aux_df = aux_df.append(df_notsafe_p, ignore_index=True)
  
  df_presence_safe[str_name] = data_p_safe_median
  df_presence_notsafe[str_name] = data_p_notsafe_median
  df_presence_global[str_name] = aux_df.mean()

df_presence_global=df_presence_global.assign(m=abs(df_presence_global.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_presence_global_aux = df_presence_global.copy()

df_presence_safe=df_presence_safe.assign(m=abs(df_presence_safe.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_presence_notsafe=df_presence_notsafe.assign(m=abs(df_presence_notsafe.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_presence_global = df_presence_global.iloc[:top_elements,:].copy()
df_presence_safe = df_presence_safe.iloc[:top_elements,:].copy()
df_presence_notsafe = df_presence_notsafe.iloc[:top_elements,:].copy()

nrow=3
ncol=1

df_list = [df_presence_global, df_presence_safe, df_presence_notsafe]
fig, axes = plt.subplots(nrow, ncol,figsize=(16,8))

for c, i in enumerate(axes):
#  axes[c].plot(x, y)
#  axes[c].set_title('cats')
    
  df_list[c].plot(ax=axes[c], kind='bar')
  ax = plt.gca()
  pos = []
  for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)
  
  axes[c].set_xticks(pos,minor=True)
  lab = []
  for i in range(len(pos)):
    l = df_list[c].columns.values[i//len(df_list[c].index.values)]
    lab.append(l)
  
  axes[c].set_xticklabels(lab,minor=True)
  axes[c].tick_params(axis='x', which='major', pad=15, size=0)
  if c==0:
    axes[c].set_ylabel('Object')
    axes[c].legend_ = None
  if c==1:
    axes[c].set_ylabel('Object safe')
    axes[c].legend_ = None
  if c==2:
    axes[c].set_ylabel('Object Not safe')
    axes[c].legend_ = None
  
  plt.setp(axes[c].get_xticklabels(), rotation=0)
  
plt.xlabel('level of presences')

plt.savefig(output_dir+"object_presence.png", bbox_inches='tight', pad_inches = 0)
plt.clf()
plt.cla()
plt.close()





df_global=df_global.assign(m=abs(df_global.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_safe_global=df_safe_global.assign(m=abs(df_safe_global.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_notsafe_global=df_notsafe_global.assign(m=abs(df_notsafe_global.mean(axis=1))).sort_values('m', ascending=False).drop('m', axis=1)

df_global = df_global.iloc[:top_elements,:].copy()
df_safe_global = df_safe_global.iloc[:top_elements,:].copy()
df_notsafe_global = df_notsafe_global.iloc[:top_elements,:].copy()

nrow=3
ncol=1
df_list = [df_global, df_safe_global, df_notsafe_global]
fig, axes = plt.subplots(nrow, ncol,figsize=(16,8))

for c, i in enumerate(axes):
#  axes[c].plot(x, y)
#  axes[c].set_title('cats')
    
  df_list[c].plot(ax=axes[c], kind='bar')
  ax = plt.gca()
  pos = []
  for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)
  
  axes[c].set_xticks(pos,minor=True)
  lab = []
  for i in range(len(pos)):
    l = df_list[c].columns.values[i//len(df_list[c].index.values)]
    lab.append(l)
  
  axes[c].set_xticklabels(lab,minor=True)
  axes[c].tick_params(axis='x', which='major', pad=15, size=0)
  if c==0:
    axes[c].set_ylabel('Object')
    axes[c].legend_ = None
  if c==1:
    axes[c].set_ylabel('Object safe')
    axes[c].legend_ = None
  if c==2:
    axes[c].set_ylabel('Object Not safe')
    axes[c].legend_ = None
  
  plt.setp(axes[c].get_xticklabels(), rotation=0)
  
plt.xlabel('level of importances')

plt.savefig(output_dir+"object_importance.png", bbox_inches='tight', pad_inches = 0)
plt.clf()
plt.cla()
plt.close()


df_global.plot(kind='bar', figsize=(16,8))
ax = plt.gca()
pos = []
for bar in ax.patches:
  pos.append(bar.get_x()+bar.get_width()/2.)

ax.set_xticks(pos,minor=True)
lab = []
for i in range(len(pos)):
  l = df_global.columns.values[i//len(df_global.index.values)]
  lab.append(l)

ax.set_xticklabels(lab,minor=True)

ax.tick_params(axis='x', which='major', pad=20, size=0)
plt.setp(ax.get_xticklabels(), rotation=0)
plt.xlabel('level of importances')
plt.ylabel('Object')
plt.tight_layout()

plt.savefig(output_dir+"object_importance__.png", bbox_inches='tight', pad_inches = 0)
plt.clf()
plt.cla()
plt.close()

## Presence

df_presence_global_new = df_presence_global_aux.loc[df_global.index.values]

df_presence_global_new.plot(kind='bar', figsize=(16,8))
ax = plt.gca()
pos = []
for bar in ax.patches:
  pos.append(bar.get_x()+bar.get_width()/2.)

ax.set_xticks(pos,minor=True)
lab = []
for i in range(len(pos)):
  l = df_presence_global_new.columns.values[i//len(df_presence_global_new.index.values)]
  lab.append(l)

ax.set_xticklabels(lab,minor=True)

ax.tick_params(axis='x', which='major', pad=20, size=0)
plt.setp(ax.get_xticklabels(), rotation=0)
plt.xlabel('level of importances')
plt.ylabel('Object')
plt.tight_layout()

plt.savefig(output_dir+"object_presence__.png", bbox_inches='tight', pad_inches = 0)
plt.clf()
plt.cla()
plt.close()


nrow=2
ncol=1
df_list = [df_global, df_presence_global_new]
fig, axes = plt.subplots(nrow, ncol,figsize=(16,8))

for c, i in enumerate(axes):
#  axes[c].plot(x, y)
#  axes[c].set_title('cats')
    
  df_list[c].plot(ax=axes[c], kind='bar')
  ax = plt.gca()
  pos = []
  for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)
  
  axes[c].set_xticks(pos,minor=True)
  lab = []
  for i in range(len(pos)):
    l = df_list[c].columns.values[i//len(df_list[c].index.values)]
    lab.append(l)
  
  axes[c].set_xticklabels(lab,minor=True)
  axes[c].tick_params(axis='x', which='major', pad=15, size=0)
  if c==0:
    axes[c].legend_ = None
    axes[c].set_ylabel('Object Importance')
  if c==1:
    axes[c].set_ylabel('Object Presence')
  
  plt.setp(axes[c].get_xticklabels(), rotation=0)
  
plt.xlabel('objects')

plt.savefig(output_dir+"object_importance_presence.png", bbox_inches='tight', pad_inches = 0)
plt.clf()
plt.cla()
plt.close()
