#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from joblib import dump, load

from utils import verifyDir
from utils.datasets import evalClass, getClassSplit, getScores
from utils.libsvm import getClassifier, getClassMetrics

from matplotlib.colors import Normalize

output_dir = "features/"
root_dir = "outputs/detector/"

verifyDir(output_dir)
verifyDir(root_dir)

# GETTING AREAS FROM GRAFFITI AND GARBAGE DETECTORS

garbage_dir = root_dir + "garbage/"
graffiti_dir = root_dir + "graffiti/"

garbage_df = pd.read_csv(garbage_dir+"garbage_segmented.csv")
graffiti_df = pd.read_csv(graffiti_dir+"graffiti_detected.csv")
graffiti_df.drop(columns=['graffiti'], inplace=True)
graffiti_df.rename(columns = {'graffiti_mask':'graffiti'}, inplace=True)
print("garbage:", garbage_df.shape)
print("graffiti:", graffiti_df.shape)

df = pd.merge(garbage_df, graffiti_df)
df.loc[:,"garbage_bag":]=df.loc[:,"garbage_bag":]*100 ######################################
print("Merging garbage and graffiti:", df.shape, "\n")

# COMPARING OBJECTS AND SEGMENTED AREAS

segmented_dir = "outputs/sceneparser/"
objects_dir = root_dir + "objects/"

segmented_df = pd.read_csv(segmented_dir+"objects_segmented.csv")
print("segmented raw:", segmented_df.shape)

objects_df = pd.read_csv(objects_dir + "objects_segmented.csv")
objects_df.loc[:,"person":]=objects_df.loc[:,"person":]*100 ################################
print("objects raw:", objects_df.shape)

# OBTAIN REPEATED COLUMNS

seg_repeated_df = segmented_df[objects_df.columns & segmented_df.columns].copy()
repeated_columns = seg_repeated_df.columns[1:].copy()
seg_names = {k:k+"_seg" for k in seg_repeated_df.columns.values[1:]}
seg_repeated_df.rename(columns = seg_names, inplace=True)
#print(seg_repeated_df)

obj_repeated_df = objects_df[objects_df.columns & segmented_df.columns].copy()
obj_names = {k:k+"_dect" for k in obj_repeated_df.columns.values[1:]}
obj_repeated_df.rename(columns = obj_names, inplace=True)
#print(obj_repeated_df)

objects_df.drop(columns=repeated_columns, inplace=True)
segmented_df.drop(columns=repeated_columns, inplace=True)
print("objects:", objects_df.shape)
print("segmented:", segmented_df.shape)

df = pd.merge(df, segmented_df)
df = pd.merge(df, objects_df)
print("Merging segmented and detected:", df.shape, "\n")

print("segmented repeated:",seg_repeated_df.shape)
print("objects repeated:",obj_repeated_df.shape)
df = pd.merge(df, seg_repeated_df)
df = pd.merge(df, obj_repeated_df)
print("Merging segmented and detected repeated elements:", df.shape, "\n")

# READING DATASET FROM BOSTON

segmented_df = pd.read_csv(segmented_dir+"objects_segmented.csv")

pp_data = getScores(["placepulse_1", "Boston", "safety"])
d_class = {"ID": pp_data[0], "y": pp_data[1]}
boston_df = pd.DataFrame(data=d_class)

dataset_df = pd.merge(segmented_df, boston_df)
dataset_df.sort_values("y", ascending=False, inplace=True)
print("dataset raw:", dataset_df.shape)

# FILTERING

dataset_df = dataset_df.loc[:, (dataset_df != 0).any(axis=0)]
print("dataset filter:", dataset_df.shape)
print(dataset_df)

dataset_df.to_csv(output_dir+'segmented_df.csv', index=False)

