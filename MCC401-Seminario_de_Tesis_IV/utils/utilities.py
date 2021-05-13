#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split

from utils.detectors import getTextsDetection
from utils.datasets import getScores, evalClass, getImageName
from utils.downloaders import is_valid_image

import numpy
import sys

def loadData(pp_data,  num_class, delta_i, img_dims=[224, 224], type_train="classification", img_dir="data/images/pp1/", year="2019"):
  images_list = pp_data[0]
  real_score = pp_data[1]
  img_list, scores = evalClass(images_list, real_score, delta_i)
  print("Load X and Y into file_list")
  files = open("dcnn/scores_"+str(delta_i)+".txt", "w")
  # number of null images
  nulls = 0
  total = 0
  
  # define X , Y
  X = []
  Y = []
  
  slen = len(scores)
  
  for i in range(slen):
    img_name = img_dir+year+"/"+str(img_list[i])+".jpg"
    if not is_valid_image(img_name):
      nulls +=1
      continue
    else:
      aux_image = load_img(img_name, target_size=(img_dims[0], img_dims[1]))
      X.append(img_to_array(aux_image))

      value = scores[i]
      if type_train=="regression":
        value = float(real_score[i])
      Y.append(value)
    files.write(img_name+" "+str(scores[i])+" "+str(real_score[i])+"\n")
    total +=1
  files.close()
  print("X, Y, file_list Done")
  print("total number of images: " + str(total))
  print("number of null images: " + str(nulls))
  X = numpy.array(X)
  Y = numpy.array(Y)
  
  if type_train=="classification" or type_train=="class":
    Y = to_categorical(Y, num_classes=num_class)
  print("X shape: ", X.shape)
  print("Y shape: ", Y.shape)
  return X, Y

def prepareData(metadata, num_class, delta_i, img_dims=[224, 224], type_train="classification", img_dir="data/images/pp1/", year="2019"):
  print("Preparing data ...")
  ## Exist file ?
  pp_data = getScores(metadata)
  ## Get data X - Y
  return loadData(pp_data, num_class, delta_i, img_dims, type_train, img_dir, year)

def divideData(X, Y, train_size_split=0.20, test_size_split=0.35):
  # Divide Data
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=train_size_split, random_state=0)
  X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, shuffle=True, test_size=test_size_split, random_state=0)
  print("Dividing data ... done")
  return X_train, X_test, X_val, Y_train, Y_test, Y_val

def getData(metadata, num_class, delta_i, img_dims=[224, 224], type_train="classification", train_size_split=0.20, test_size_split=0.35, img_dir="data/images/pp1/", year="2019"):
  X, Y = prepareData(metadata, num_class, delta_i, img_dims, type_train, img_dir, year)
  print("Preparing data ... done")
  return divideData(X, Y, train_size_split, test_size_split)
