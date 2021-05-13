#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Layer

class LRN(Layer):
  def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
    self.n = n
    self.alpha = alpha
    self.beta = beta
    self.k = k
    super(LRN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.shape = input_shape
    super(LRN, self).build(input_shape)

  def call(self, x, mask=None):
    if K.image_dim_ordering == "th":
      _, f, r, c = self.shape
    else:
      _, r, c, f = self.shape
    half_n = self.n // 2
    squared = K.square(x)
    pooled = K.pool2d(squared, (half_n, half_n), strides=(1, 1),
                     border_mode="same", pool_mode="avg")
    if K.image_dim_ordering == "th":
      summed = K.sum(pooled, axis=1, keepdims=True)
      averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=1)
    else:
      summed = K.sum(pooled, axis=3, keepdims=True)
      averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=3)
    denom = K.pow(self.k + averaged, self.beta)
    return x / denom
  
  def get_output_shape_for(self, input_shape):
    return input_shape



def global_average_pooling(x):
  return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
  return input_shape[0:2]
