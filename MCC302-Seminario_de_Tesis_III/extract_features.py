#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from keras import backend as K
import numpy as np
from joblib import dump, load

from utils import verifyFile, verifyDir
from utils.datasets import getScores
from utils.downloaders import is_valid_image
from utils.networks import clearSession, getModel, verifyDevices, getPreprocess

from keras.models import Model
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout

years = ["2011", "2013", "2019"]
metrics = ["safety", "wealthy", "uniquely"]
cities = ["Boston", "New York City"]

clearSession()
verifyDevices("gpu")

features_dir = "features/"
verifyDir(features_dir)

models = ["vgg16_places", 'vgg16_gap_places', 'vgg16', 'vgg16_gap']

def getLayerOutput(name_model="vgg16"):
  if name_model=="vgg16_gap":
    model = getModel(name_model="vgg16", include_top=False, weights='imagenet')
    cnn = model.output
    cnn = GlobalAveragePooling2D(name='GAP')(cnn)
    dcnn_model = Model(inputs=model.input, outputs=cnn, name="famv-classifier")
    get_output = K.function([dcnn_model.layers[0].input], [dcnn_model.layers[-1].output])
  elif name_model == "vgg16":
    dcnn_model = getModel(name_model="vgg16", include_top=True, weights='imagenet')
    get_output = K.function([dcnn_model.layers[0].input], [dcnn_model.layers[-2].output])
  elif name_model == "vgg16_gap_places":
    dcnn_model = getModel(name_model="vgg16_places365")
    get_output = K.function([dcnn_model.layers[0].input], [dcnn_model.layers[-1].output])
  elif name_model == "vgg16_places":
    dcnn_model = getModel(name_model="vgg16_places365", include_top=True)
    get_output = K.function([dcnn_model.layers[0].input], [dcnn_model.layers[-2].output])
  
  dcnn_model.summary()
  
  return get_output

for model_name in models:
  get_layer_output = getLayerOutput(name_model=model_name)
  name_dir = features_dir + model_name
  for city in cities:
    for metric in metrics:
      print("preparing {}-{} data ...".format(city, metric))
      
      pp_data = getScores(["placepulse_1", city, metric])

      images_list = pp_data[0]
      scores_list = pp_data[1]
      
      leng = len(images_list)
      
      for year in years:
        images_dir = "data/images/pp1/"+year+"/"
        X, X_img, y = [], [], []
        
        for i in range(0, leng):
          img = images_list[i]
          img_path = images_dir+str(img)+".jpg"
          if not is_valid_image(img_path):
            continue
          
          # loadImage
          orig, preprocess, img_name, img_dims, img_orig_dims = getPreprocess(img_path)
          x_aux = get_layer_output([preprocess[np.newaxis,...]])[0]
          X.append(x_aux)
          X_img.append(preprocess)
          y.append(scores_list[i])
      
        X = np.asarray(X)
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples, nx*ny))
        X_img = np.asarray(X_img)
        y = np.asarray(y)
        print(X_img.shape)
        print(X.shape)
        print(y.shape)

        #dump(X_img, features_dir+"X_"+year+"_"+city+"_"+metric+".joblib")
        #dump(y, features_dir+"y_"+year+"_"+city+"_"+metric+".joblib")
        
        #np.savetxt(name_dir+"_X_"+year+"_"+city+"_"+metric+".csv", X, delimiter=",")
        #np.savetxt(name_dir+"_y_"+year+"_"+city+"_"+metric+".csv", y, delimiter=",")
        
        import pandas

        variables = ["x"+str(i+1) for i in range(X.shape[1])]    
        X_data = pandas.DataFrame(X, columns= variables)
        y_data = pandas.DataFrame(y, columns=["y"])
        id_data = pandas.DataFrame(images_list, columns=["ID"])
        data = pandas.concat([id_data, X_data, y_data], axis=1)
        data.to_csv(name_dir+"_"+year+"_"+city+"_"+metric+".csv", index=False)
  
  clearSession()
