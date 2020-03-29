#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from keras import backend as K
import numpy as np

from utils import verifyFile, verifyDir
from utils.datasets import getScores
from utils.downloaders import is_valid_image
from utils.networks import clearSession, getModel, verifyDevices, getPreprocess

root_path = "data/outputs/"
type_path = "classification/"
model_dir = "vgg16_64_80/"
dataset_path = "placepulse_1/"

complete_path = root_path + type_path + model_dir + dataset_path + "models/"

year="2013"

images_dir = "data/images/pp1/"+year+"/"

clearSession()
verifyDevices("gpu")

name_dir = "features/"
verifyDir(name_dir)

metrics = ["safety"]#, "wealthy", "uniquely"]
cities = ["Boston"]#, "New York City"]

def getLayerOutput(model, output_layer=-2):
  get_output = K.function([model.layers[0].input], [model.layers[output_layer].output])
  return get_output

model_classifier = getModel(name_model="vgg16", include_top=True)
get_layer_output = getLayerOutput(model_classifier)

for city in cities:
  for metric in metrics:
    print("preparing {}-{} data ...".format(city, metric))
    
    pp_data = getScores(["placepulse_1", city, metric])

    images_list = pp_data[0]
    scores_list = pp_data[1]
    
    leng = len(images_list)
    
    X, y = [], []
    
    for i in range(0, leng):
      img = images_list[i]
      img_path = images_dir+str(img)+".jpg"
      if not is_valid_image(img_path):
        continue
      
      # loadImage
      orig, preprocess, img_name, img_dims, img_orig_dims = getPreprocess(img_path)
      x_aux = get_layer_output([preprocess[np.newaxis,...]])[0]
      X.append(x_aux)
      y.append(scores_list[i])
  
    X = np.asarray(X)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx*ny))
    y = np.asarray(y)
    print(X.shape)
    print(y.shape)

    np.savetxt("X_"+city+"_"+year+".csv", X, delimiter=",")
    np.savetxt("y_"+city+"_"+year+".csv", y, delimiter=",")
    
    import pandas

    variables = ["x"+str(i+1) for i in range(X.shape[1])]    
    X_data = pandas.DataFrame(X, columns= variables)
    y_data = pandas.DataFrame(y, columns=["y"])
    data = pandas.concat([X_data, y_data], axis=1)
    data.to_csv("vgg16_"+city+".csv", index=False)
    
