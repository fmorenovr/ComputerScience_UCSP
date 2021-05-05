#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .session import verifyDevices, clearSession, deprocess_image, clearModel
from .models import getModel, getPreprocess, loadModel, getPreprocess, freezeLayers, unFreezeLayers, getOptimizer, getLearningRate, getPooling, getPredictions
from .metrics import *

#def getPreprocess(img, img_dims=[224, 224], name_model="vgg16"):
#def getPredictions(img, model, top_labels=1000):

def getImgPreproPreds(img, model, top_labels, img_dims=[224, 224], name_model="vgg16"):
  orig, preprocess, img_name, img_dims, img_orig_dims = getPreprocess(img, img_dims, name_model)
  prob_preds, labels_preds, vect_probs = getPredictions(preprocess, model, top_labels)

  imgInfo = [orig, preprocess, img_name, img_dims, img_orig_dims]
  predInfo = [prob_preds, labels_preds, name_model, vect_probs]
  return imgInfo, predInfo
  
  
