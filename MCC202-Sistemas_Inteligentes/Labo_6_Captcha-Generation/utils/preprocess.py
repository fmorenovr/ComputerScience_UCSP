#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def lrelu(x, alpha=0.3):
  return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob, n_latent):
  activation = lrelu
  with tf.variable_scope("encoder", reuse=None):
    X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
    x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.contrib.layers.flatten(x)
    mn = tf.layers.dense(x, units=n_latent)
    sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
    epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
    z  = mn + tf.multiply(epsilon, tf.exp(sd))
    return z, mn, sd

def decoder(sampled_z, keep_prob, inputs_decoder, reshaped_dim):
  with tf.variable_scope("decoder", reuse=None):
    x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
    x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
    x = tf.reshape(x, reshaped_dim)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
    
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
    img = tf.reshape(x, shape=[-1, 28, 28])
    return img
