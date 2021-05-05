#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda
from keras.models import model_from_json, load_model
from keras.preprocessing.image import load_img, img_to_array

from utils.datasets import getImageName, getPredictedCategories

from .session import getInputShape
from .resnet50 import *
from .vgg16 import *
from .vgg19 import *
from .vgg16_places365 import *
from .vgg16_places1365 import *

def loadModel(model_path):
  model = load_model(model_path)
  return model

def freezeLayers(model, pos_layer=None):
  # Freeze the layers
  if pos_layer==None:
    for layer in model.layers:
      layer.trainable = False
  else:
    for layer in model.layers[:pos_layer]:
      layer.trainable = False
  return model

def unFreezeLayers(model, pos_layer=None):
  # Freeze the layers
  if pos_layer==None:
    for layer in model.layers:
      layer.trainable = True
  else:
    for layer in model.layers[pos_layer:]:
      layer.trainable = True
  return model

def getModel(img_dims=[224, 224], name_model="famv", include_top=False, weights='imagenet', freeze_layers=None, log=False):
  input_shape = getInputShape(img_dims)
  print("Selecting model ...")
  ## MODEL
  if name_model == "vgg16":
    pre_model = vgg16(input_shape, include_top, weights)
    pre_model = freezeLayers(pre_model, freeze_layers)
  elif name_model == "vgg19":
    pre_model = vgg19(input_shape, include_top, weights)
    pre_model = freezeLayers(pre_model, freeze_layers)
  elif name_model == "resnet50":
    pre_model = resnet50(input_shape, include_top, weights)
    pre_model = freezeLayers(pre_model, freeze_layers)
  elif name_model == "vgg16_places365":
    pre_model = VGG16_Places365(input_shape, include_top, pooling="avg")
    pre_model = freezeLayers(pre_model, freeze_layers)
  elif name_model == "vgg16_places1365":
    pre_model = VGG16_Hybrid_1365(input_shape, include_top, pooling="avg")
    pre_model = freezeLayers(pre_model, freeze_layers)
  else:
    pre_model = load_model(weights)

  if log:
    for layer in pre_model.layers:
      print(layer, layer.trainable)
    pre_model.summary()
  print("Done")
  return pre_model
  
def getLearningRate(alpha=1, decay_name="optimal"):
  if decay_name=="optimal":
    def getLR(alpha):
      #typw = np.sqrt(1.0 / np.sqrt(alpha))
      # computing eta0, the initial learning rate
      #initial_eta0 = typw / max(1.0, loss.dloss(-typw, 1.0))
      # initialize t such that eta at first sample equals eta0
      optimal_init = 1.0 / (np.sqrt(1.0 / np.sqrt(alpha)) * alpha)
      return optimal_init
      
    def decayOptimal(epoch, lr):
      if epoch<2:
        return getLR(alpha)
      else:
        return 1.0 / (alpha * (getLR(alpha) + epoch - 1))
    
    return decayOptimal
    
  else:
    def decayStep(epoch, lr):
      decay_rate = 0.85
      decay_step = 6
      if epoch%decay_step == 0 and epoch>24:
        lr = lr*decay_rate
        if lr < 35e-5:
          lr = 0.0005
        return lr
      return lr
    return decayStep

def getPreprocess(img, img_dims=[224, 224], name_model="famv"):
  img_name = getImageName(img)
  input_shape = getInputShape(img_dims)
  orig_shapes = load_img(img)
  orig_shapes = img_to_array(orig_shapes)
  
  ## MODEL
  if name_model == "vgg16":
    orig, preprocess = vgg16LoadImage(img, img_dims)
  elif name_model == "vgg19":
    orig, preprocess = vgg19LoadImage(img, img_dims)
  elif name_model == "resnet50":
    orig, preprocess = resnet50LoadImage(img, img_dims)
  else:
    orig, preprocess = loadImage(img, img_dims)
  
  return orig, preprocess, img_name, img_dims, [orig_shapes.shape[0], orig_shapes.shape[1]]

def loadImage(path, img_dims=[224, 224]):
  """Load and preprocess image."""
  orig = load_img(path, target_size=img_dims)
  preprocess = img_to_array(orig)
  return orig, preprocess

def getOptimizer(name_opt, lr):
  if name_opt == "sgd":
    opt = SGD(lr=lr, momentum=0.9)
  elif name_opt == "adam":
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  elif name_opt == "rmsprop":
    opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
  elif name_opt == "adadelta":
    opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
  else:
    opt = SGD(lr=lr, momentum=0.9)
  return opt

def getPooling(cnn, pooltype="avg"):
  if pooltype=="avg":
    cnn = GlobalAveragePooling2D()(cnn)
  elif pooltype =="max":
    cnn = GlobalMaxPooling2D()(cnn)
  else:
    cnn = GlobalAveragePooling2D()(cnn)
  return cnn

# get prediction from image input
def getPredictions(img, model, top_labels=1000):
  # y_predict or probability vector (per each class)
  vect_probs = model.predict(img[np.newaxis,...], verbose=1)
  # indexes of the first top probs (max values)
  max_top_index_vect_probs = np.argsort(vect_probs.flatten())[::-1][:top_labels]
  # values inside indexes obtained before
  prob_preds = [obj[max_top_index_vect_probs[i]] for i, obj in enumerate(vect_probs)] #vect_probs[max_top_index_vect_probs]
  # print predictions
  labels_preds = getPredictedCategories(vect_probs, max_top_index_vect_probs)
  
  return prob_preds, labels_preds, vect_probs

def get_output_layer(model, layer_name):
  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  layer = layer_dict[layer_name]
  return layer

def saveJSONModel(new_model, name_json, namefile):
  print("Saving model to disk ...")
  # serialize model to JSON
  model_json = new_model.to_json()

  name_weights = "models/"+namefile+".h5"

  with open(name_json, "w") as json_file:
    json_file.write(model_json)

  # serialize weights to HDF5
  new_model.save_weights(name_weights)
  print("Done")

def loadJSONModel(name_json, name_weights, lr):
  print("Loading model from disk ...")
  # load json and create model
  
  with open(name_json, 'r') as json_file:
    loaded_model = model_from_json(json_file.read())
  
  # load weights into new model
  loaded_model.load_weights(name_weights)
  print("Done")
  
  print("Compiling model ...")
  sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
  adam = Adam(lr=lr)
  loaded_model.compile(sgd, loss='categorical_crossentropy', metrics=['acc', 'mse'])
  print("Done")
  return loaded_model

