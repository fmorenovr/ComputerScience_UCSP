#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import json
import numpy

root_path = "data/categories/"

def getPredictedCategories(vect_probs, max_top_index_vect_probs, dataset="imagenet", num_cat=1000):
  classlabel = getCategories(dataset, num_cat)
  print('--PREDICTED SCENE/OBJECTS CATEGORIES:')
  # output the prediction
  labels_preds = []
  for i, idx in enumerate(max_top_index_vect_probs):
    labels_preds.append(classlabel[max_top_index_vect_probs[i]])
    print("Top {} predicted class:  Pr(Class={:18} [index={}])={:5.3f}".format(i + 1, classlabel[idx], idx, vect_probs[0, idx]))
  return labels_preds

def getCategories(dataset="imagenet", num_cat=1000):
  classlabel = []
  if dataset == "imagenet":
    classlabel = ImageNet()
  elif dataset == "places":
    classlabel = Places(num_cat)
  elif dataset == "sunattr":
    classlabel = SUNAttr()
  return classlabel

# ImageNet
def ImageNet():
  classlabel  = []
  CLASS_INDEX = json.load(open(root_path+"imagenet/imagenet_class_index.json"))
  for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
  print("N of class={}".format(len(classlabel)))
  return classlabel

# Places
def Places(num_cat=365):
  classlabel = []
  if num_cat == 205:
    CLASS_INDEX = root_path+"places/places/places205.names"
    with open(CLASS_INDEX) as class_file:
      for line in class_file:
        classlabel.append(line.strip().split(' ')[0][3:])
    print("N of class={}".format(len(classlabel)))
  elif num_cat == 365:
    CLASS_INDEX = root_path+"places/places2/places365.names"
    with open(CLASS_INDEX) as class_file:
      for line in class_file:
        classlabel.append(line.strip().split(' ')[0][3:])
    print("N of class={}".format(len(classlabel)))
  elif num_cat == 1365:
    CLASS_INDEX = root_path+"places/places2/hybrid1365.names"
    counter = 0
    with open(CLASS_INDEX) as class_file:
      for line in class_file:
        if counter <=999:
          tmp = line[9:]
          if 0 <= counter <= 9:
              tmp = tmp[:-2]
          elif 10 <= counter <= 99:
              tmp = tmp[:-3]
          elif 100 <= counter <= 999:
              tmp = tmp[:-4]
          classlabel.append(tmp)
        else:
          classlabel.append(line.strip().split(' ')[0][3:])
        counter +=1
    print("N of class={}".format(len(classlabel)))
  elif num_cat == 2: # input - output
    file_name_IO = root_path+"places/places2/Indoor-Outdoor_places365.names"
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    classlabel = numpy.array(labels_IO)
  return classlabel

def SUNAttr():
  file_name_attribute = root_path+"SUN_Attributes/sunattributes.names"
  labels_attribute = []
  with open(file_name_attribute) as f:
    lines = f.readlines()
    labels_attribute = [item.rstrip() for item in lines]
  return labels_attribute
