#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
CLASSES = 10

def load_images(dataname):
  #Sets the threshold for what messages will be logged.
  old_v = tf.logging.get_verbosity()
  # able to set the logging verbosity to either DEBUG, INFO, WARN, ERROR, or FATAL. Here its ERROR
  tf.logging.set_verbosity(tf.logging.ERROR)
  mnist = input_data.read_data_sets(dataname)
  #in the end
  tf.logging.set_verbosity(old_v)
  return mnist

def getImagefromDataset(num, mnist):
  xs, ys = mnist.train.next_batch(batch_size=BATCH_SIZE)
  xs_l = []
  for i in range(BATCH_SIZE):
    if np.argmax(ys[i]) == num:
      xs_l.append(xs[i])
  return xs_l

def getZ(mu, sigma, sess, x_dim, X):
  x = tf.placeholder(tf.float32, shape=[None, x_dim])
  mutf = tf.convert_to_tensor(mu, dtype=tf.float32)
  sigmatf = tf.convert_to_tensor(sigma, dtype=tf.float32)
  z_mean_, z_log_sigma_sq = sess.run((mutf, sigmatf), feed_dict={x: X})  
  return z_mean_, z_log_sigma_sq

def getParams(num, mu, sigma, sess, x_dim, mnist):
  imgsX    = getImagefromDataset(num, mnist)
  k       = len(imgsX)
  if k == 0:
    return mu, sigma
  mu_t, sig_t = 0.0, 0.0
  for i in range(k):   
    mu1, sig1 = getZ(mu, sigma, sess, x_dim, [imgsX[i]])
    #print(mu1, sig1)
    mu_t   += mu1
    sig_t  += np.sqrt(np.exp(sig1))
  return mu_t/k, sig_t/k

def generateParameters(mu, sigma, sess, x_dim, mnist):
  params  = []
  for i in range(CLASSES):
    mu1, sigma1 = getParams(i, mu, sigma, sess, x_dim, mnist)
    params.append([mu1, sigma1])
  return params

def getImgFromVAE(mu, sigma, sess, x_dim, mnist, inum, n_latent, sampled, dec, keep_prob):
  M = generateParameters(mu, sigma, sess, x_dim, mnist)
  randoms = np.array([np.random.normal(0, 1, n_latent) for _ in range(1)])
  z_ = M[inum][0] + M[inum][1] * randoms
  z_mu = np.array([z_[0]]*BATCH_SIZE)
  imgs = sess.run(dec, feed_dict = {sampled: z_mu, keep_prob: 1.0})
  imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
  img = imgs[0]
  '''
  plt.figure(figsize=(1, 1))
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  plt.show()
  '''
  return img

def getListOfNumbers(mu, sigma, sess, x_dim, mnist, sNUM, n_latent, sampled, dec, keep_prob):
  res = []
  for i in range(len(sNUM)):
    img = getImgFromVAE(mu, sigma, sess, x_dim, mnist, int(sNUM[i]), n_latent, sampled, dec, keep_prob)
    res.append(img)
  return res
