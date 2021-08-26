import cv2
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from joblib import dump, load

from utils import verifyDir
from utils.datasets import evalClass, getClassSplit, getScores
from utils.libsvm import getClassifier, getClassMetrics

from matplotlib.colors import Normalize

# CLASSIFICATION

input_dir = "features/"

output_dir_ = "to_test/"

psp_df = pd.read_csv(input_dir+'objects_segmented.csv')
psp_df["ID"] = psp_df["ID"].apply(lambda x: x.split("_")[1])
#print(psp_df)

psp_df = psp_df.loc[:, (psp_df != 0).any(axis=0)]

# Total
X_df = psp_df.iloc[:, 1:]
X_df = X_df[np.sort(X_df.columns)]
X_df[X_df > 0] = 1

sum_df = X_df.sum()/len(X_df)

ax = sum_df.plot(kind="bar", figsize=(16,8), ylabel="% Presence", rot=90, ylim=(0.0,1.0), yticks=list(np.arange(0,1,0.10)))
plt.title(f"Boston objects presence")
plt.grid(True)

plt.savefig("Total_psp.png", bbox_inches='tight', pad_inches = 0.5)
plt.clf()
plt.cla()
plt.close()


output_dir_ = "to_test_pp2/"

psp_df = pd.read_csv(input_dir+'Boston_deeplab_xception.csv', sep=";")

psp_df.drop(columns=["no class", "uniquely", "wealthy", "safety"], inplace=True)

psp_df["ID"] = psp_df["ID"].apply(lambda x: x.split("_")[1])
#print(psp_df)

psp_df = psp_df.loc[:, (psp_df != 0).any(axis=0)]

# Total
X_df = psp_df.iloc[:, 1:]
X_df = X_df[np.sort(X_df.columns)]
X_df[X_df > 0] = 1

sum_df = X_df.sum()/len(X_df)

ax = sum_df.plot(kind="bar", figsize=(16,8), ylabel="% Presence", rot=90, ylim=(0.0,1.0), yticks=list(np.arange(0,1,0.10)))
plt.title(f"Boston objects presence")
plt.grid(True)

plt.savefig("Total.png", bbox_inches='tight', pad_inches = 0.5)
plt.clf()
plt.cla()
plt.close()
