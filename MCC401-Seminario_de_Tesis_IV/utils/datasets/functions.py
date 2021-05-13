#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np
import collections
import zipfile

def setPathName(name):
  aux = str.split(name, "id")
  aux1 = str.split(aux[1], "_")
  name = "data"+aux[0]+aux1[1]+".jpg"
  return name

# Read placepulse_2.zip
def read_zip_file(filepath):
  data = zipfile.ZipFile(filepath)
  return data

def newRange(OldValues, OldMin, OldMax, NewMin=0, NewMax=10):
  NewValues = []
  OldRange = (OldMax - OldMin)
  for OldValue in OldValues:
    NewRange = NewMax - NewMin
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    NewValues.append(NewValue)
  return np.array(NewValues)

def setName(name):
  aux = str.split(name, "id")
  aux1 = str.split(aux[1], "_")
  name = aux1[1]+".jpg"
  return name

def getLabels(y, delta):
  y = np.asarray([1 if a_ > 5 else 0 for a_ in y])
  return y

def evalClass(features, scores, delta, img_names=None, type_m="classification", log=False, dir_path="dcnn/scores_", log_count=False):
  slen = len(scores)
  val = int(delta*slen)
  if log:
    files = open(dir_path+str(delta)+".txt", "w")
  X = []
  Y = []
  for i in range(slen):
    if type_m=="classification":
      if scores[i] in scores[:val]: # from 0 until val-1 top 
        aux_Y = 1
      elif scores[i] in scores[slen-val:]: # from slen-val until slen-1 bottom
        aux_Y = 0
      else:
        continue
    else:
      if scores[i] in scores[:val]: # from 0 until val-1 top 
        aux_Y = scores[i]
      elif scores[i] in scores[slen-val:]: # from slen-val until slen-1 bottom
        aux_Y = scores[i]
      else:
        continue
    if log:
      files.write(str(img_names[i])+" "+str(scores[i])+" "+str(aux_Y)+"\n")
    aux_X = features[i]
    Y.append(aux_Y)
    X.append(aux_X)
  
  if log:
    files.close()

  counter=collections.Counter(Y)
  if log_count:
    print("scores[:val]: ", len(scores[:val]), " scores[len(scores)-val:]", len(scores[len(scores)-val:]), " len: ", len(scores))
    print(counter)
  
  return np.array(X), np.array(Y)

def getFileFormat(name):
  return str.split(name, ".")[1]

def getIdName(idname):
 strg = str.split(idname, ".")
 return strg[0]

def cleanNames(name):
  a = str.split(name, "/")
  new_name = a[len(a)-1]
  return new_name

def setPath(name):
  aux = str.split(name, "_")
  name = "./data"+aux[0]+"_"+aux[1]+"_640_420.jpg"
  return name

def getImageName(filename, extension=False):
  name = str.split(filename, "/")
  name = name[len(name)-1]
  if extension:
    return name
  else:
    return getIdName(name)

def intervals(arr, num_class):
  maxEle = np.amax(arr)
  minEle = np.amin(arr)
  interval = (maxEle - minEle)/(num_class+1)
  quantiles = []
  for i in range(num_class):
    quantiles.append(minEle + i*interval)
  quantiles.append(maxEle)
  return quantiles
