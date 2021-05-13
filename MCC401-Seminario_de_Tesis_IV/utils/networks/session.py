#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import os
import gc
import torch
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf
from tensorflow.python.client import device_lib

## Cleanning previous session
def clearSession(model=None):
  print("Free memory in GPU ...")
  try:
    del model # this is from global space - change this as you need
  except:
    pass
  gc.collect()
  torch.cuda.empty_cache()
  K.clear_session()
  print("Done")

def clearModel(model=None):
  print("Free memory in GPU ...")
  try:
    del model # this is from global space - change this as you need
  except:
    pass
  print("Done")

def setDeviceConfig(num_CPU, num_GPU):
  num_cores = int(os.cpu_count()/4)
  config = tf.ConfigProto(
              log_device_placement=False,
              intra_op_parallelism_threads=num_cores,
              inter_op_parallelism_threads=num_cores, 
              allow_soft_placement=True,
              device_count = {'CPU' : num_CPU,
                              'GPU' : num_GPU}
              )
  config.gpu_options.allow_growth=True
  session = tf.Session(config=config)
  K.set_session(session)

def verifyDevices(device="gpu"):
  clearSession()
  if device == "gpu":
    num_GPU = 1
    num_CPU = 1
    print("Verifiying GPU device ...")
    devices = device_lib.list_local_devices()
    #print(devices)
    if not 'GPU' in str(devices) or not tf.test.is_gpu_available():
      print("Dont have GPU device or you cant use your GPU for missing libs")
      print("USING CPU")
      num_CPU = 1
      num_GPU = 0
    else:
      print("USING GPU")
  else:
    num_CPU = 1
    num_GPU = 0
    print("USING CPU")
  
  setDeviceConfig(num_CPU, num_GPU)
  
  print("Done")

def getInputShape(img_dims=[224, 224]):
  img_width, img_height = img_dims[0], img_dims[1]
  if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
  else:
    input_shape = (img_width, img_height, 3)
  return input_shape

def deprocess_image(x):
  """Same normalization as in:
  https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
  """
  x = x.copy()
  if numpy.ndim(x) > 3:
    x = numpy.squeeze(x)
  # normalize tensor: center on 0., ensure std is 0.1
  x -= x.mean()
  x /= (x.std() + 1e-5)
  x *= 0.1

  # clip to [0, 1]
  x += 0.5
  x = numpy.clip(x, 0, 1)

  # convert to RGB array
  x *= 255
  #if K.image_dim_ordering() == 'th':
  #  x = x.transpose((1, 2, 0))
  
  if K.image_data_format() == 'channels_first':
    x = x.transpose((1, 2, 0))
  x = numpy.clip(x, 0, 255).astype('uint8')
  return x
