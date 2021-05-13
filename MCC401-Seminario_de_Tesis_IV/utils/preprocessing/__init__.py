from .preprocess_pp1 import preprocess_pp1
from .preprocess_pp2 import preprocess_pp2

from scipy.io import savemat, loadmat
from numpy import genfromtxt
import pandas as pd
import numpy as np

def getFeatures(features_name="vgg16_gap", city="Boston", metric="safety", year="2011", features_dir="features/"):
  
  name_file = features_dir + features_name + "_"+year+"_"+city+"_"+metric+".csv"
  
  if features_name == 'vgg16':
    print("Selected VGG16")
    df = pd.read_csv(name_file)
    X_ = df.iloc[:, 1:-1].values
    Y_ = df["y"].values
    print("X_vgg16.shape: ", X_.shape)
    print("y_vgg16.shape: ", Y_.shape)
    
  elif features_name == "vgg16_places":
    print("Selected VGG16 Places")
    df = pd.read_csv(name_file)
    X_ = df.iloc[:, 1:-1].values
    Y_ = df["y"].values
    print("X_vgg16_places.shape: ", X_.shape)
    print("y_vgg16_places.shape: ", Y_.shape)
    
  elif features_name == 'vgg16_gap':
    print("Selected VGG16 GAP")
    df = pd.read_csv(name_file)
    X_ = df.iloc[:, 1:-1].values
    Y_ = df["y"].values
    print("X_vgg16_gap.shape: ", X_.shape)
    print("y_vgg16_gap.shape: ", Y_.shape)
    
  elif features_name == 'vgg16_gap_places':
    print("Selected VGG16 GAP")
    df = pd.read_csv(name_file)
    X_ = df.iloc[:, 1:-1].values
    Y_ = df["y"].values
    print("X_vgg16_gap_places.shape: ", X_.shape)
    print("X_vgg16_gap_places.shape: ", Y_.shape)
    
  elif features_name == 'gist':
    print("Selected GIST")
    if metric=="safety":
     metric="safer"
    elif metric=="uniquely":
     metric="unique"
    elif metric=="wealthy":
     metric="upperclass"
    
    name_file = features_dir + features_name + "_"+year+"_"+city+"_"+metric+".mat"
    mat = loadmat(name_file)
    gist = mat["gist_feature_matrix"]

    #image_names = mat["image_list"][0]
    scores = mat["scores"][0]
    
    Y_scores = np.asarray(scores)
    
    sort_index = np.flip(np.argsort(Y_scores))
    Y_scores = Y_scores[sort_index]

    X_ = np.asarray(gist)[sort_index]
    Y_ = Y_scores.copy()
    print("X_gist.shape: ", X_.shape)
    print("y.shape: ", Y_.shape)
    
  elif features_name == 'fisher':
    print("Selected FISHER")
    if metric=="safety":
     metric="safer"
    elif metric=="uniquely":
     metric="unique"
    elif metric=="wealthy":
     metric="upperclass"
     
    name_file = features_dir + features_name + "_"+year+"_"+city+"_"+metric+".mat"
    mat = loadmat(name_file)
    fisher = mat["fisher_feature_matrix"]

    #image_names = mat["image_list"][0]
    scores = mat["scores"][0]
    
    Y_scores = np.asarray(scores)
    
    sort_index = np.flip(np.argsort(Y_scores))
    Y_scores = Y_scores[sort_index]
    
    X_ = np.asarray(fisher)[sort_index]
    Y_ = Y_scores.copy()
    print("X_fisher.shape: ", X_.shape)
    print("y.shape: ", Y_.shape)
    
  else:
    print("Selected VGG16")
    df = pd.read_csv(name_file)
    X_ = df.iloc[:, 1:-1].values
    Y_ = df["y"].values
    print("X_vgg16.shape: ", X_.shape)
    print("y_vgg16.shape: ", Y_.shape)
  
  return X_, Y_
