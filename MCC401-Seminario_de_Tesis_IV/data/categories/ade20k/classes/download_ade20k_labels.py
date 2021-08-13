#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import os
import wget
from pandas_ods_reader import read_ods

path = "urban_perception/data/categories/ade20k/color_coding_semantic_segmentation_classes.ods"
df = read_ods(path, 1)

result = df['Name'].values
webpath = "http://scenesegmentation.csail.mit.edu/color150/"

dir_saved = "urban_perception/data/categories/ade20k/classes/"

def sliit(name):
  return str.split(name,";")

for ob in result:
  obj = sliit(ob)
  if len(obj) > 1:
    for item in obj:
      ite = item+".jpg"
      if os.path.isfile(dir_saved+ite):
        continue
      try:
        wget.download(webpath+ite)
        print("\ndownloaded", webpath+ite)
      except:
        print("\nCant download", webpath+ite, "\n")
        continue
  else:
    obj = sliit(ob)[0] + ".jpg"
    if os.path.isfile(dir_saved+obj):
      continue
    try:
      wget.download(webpath+obj)
      print("\ndownloaded", webpath+obj)
    except:
      print("\nCant download", webpath+obj,"\n")
      continue
