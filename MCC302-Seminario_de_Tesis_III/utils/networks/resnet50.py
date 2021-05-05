#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input

def resnet50(input_shape, include_top=False, weights='imagenet'):
  ## calling keras model
  pre_model = ResNet50(input_shape=input_shape, weights=weights, include_top=include_top)
  #pre_model.summary()
  return pre_model

def resnet50LoadImage(path, img_dims=[224, 224],  pre_process=True):
  """Load and preprocess image."""
  orig = load_img(path, target_size=img_dims)
  preprocess = img_to_array(orig)
  if pre_process:
    #preprocess = numpy.expand_dims(preprocess, axis=0)
    preprocess = preprocess_input(preprocess)
  return orig, preprocess
