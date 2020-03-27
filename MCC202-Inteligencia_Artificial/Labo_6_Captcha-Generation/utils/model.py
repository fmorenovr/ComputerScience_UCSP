#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#ipy = get_ipython()
#if ipy is not None:
#  ipy.run_line_magic('matplotlib', 'inline')

from utils.images import load_images, getListOfNumbers
from utils.preprocess import encoder, decoder
from utils.captchaprocess import *

EPOCHS = 30000
BATCH_SIZE = 100

DEC_IN_CHANNELS = 1
N_LATENT = 8

def generate_captcha(input_imgs):
  normList = []
  for k in range(len(input_imgs)):
    norm = cv2.normalize(input_imgs[k], None,0, 255, cv2.NORM_MINMAX)
    normList.append(norm)

  captcha =[]
  
  for i in range(len(normList)):
    res_r = rotate(normList[i])
    res_t = traslation(res_r)
    res_t_res = cv2.resize(res_t, (0,0), fx=20, fy=20) 
    new   = morphology(res_t_res)
    new   = new.astype(np.uint8)
    #plt.imshow(new)
    #plt.show()
    new_RGB = cv2.cvtColor(new, cv2.COLOR_GRAY2RGB)
    color = np.zeros((*new.shape,3), np.uint8)
    
    R     = 255#random.randint(0,255)
    G     = 255#random.randint(0,255)
    B     = 255#random.randint(0,255)
    color[:,:,:] = (R, G, B)
    ret, mask = cv2.threshold(new, 100, 255, cv2.THRESH_BINARY)
    mask_inv  = cv2.bitwise_not(mask)
    
    #bg  = np.zeros((*new.shape,3), np.uint8)
    
    img1_bg   = cv2.bitwise_and(new_RGB,new_RGB,mask = mask_inv)
    img2_fg   = cv2.bitwise_and(color,color,mask = mask)
    dst       = cv2.add(img1_bg,img2_fg)
    captcha.append(dst)
  
  chr_num = len(captcha)
  vis = np.concatenate((captcha), axis=1)
  
  lines_img      = np.zeros(vis.shape, dtype=np.uint8)
  lines_img      = addNoise(lines_img,chr_num)
  lines_img      = addLines(lines_img,chr_num)
  lines_img_blur = cv2.GaussianBlur(lines_img,(45,45),0)
  lines          = cv2.add(vis,lines_img_blur)
  #noise_img = addNoise(vis, chr_num)

  #noise_img_res = cv2.resize(noise_img, (0,0), fx=20, fy=20) 
  #lines         = addLines(noise_img,chr_num)
  lines = cv2.resize(lines,(len(input_imgs)*120, 400))
  cv2.imshow('Captcha Generate', lines)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def trainModel():
  print("\n============================================")
  print("             Preprocessing Data               ")
  print("============================================\n")

  # Load data
  mnist = load_images("MNIST_data")
  
  # Preparing Input and Output
  tf.reset_default_graph()

  X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
  Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
  Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
  keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

  x_dim = mnist.train.images.shape[1]

  reshaped_dim = [-1, 7, 7, DEC_IN_CHANNELS]
  inputs_decoder = int(49 * DEC_IN_CHANNELS / 2)

  sampled, mn, sd = encoder(X_in, keep_prob, N_LATENT)
  dec = decoder(sampled, keep_prob, inputs_decoder, reshaped_dim)

  # Loss function and gaussian distribution

  unreshaped = tf.reshape(dec, [-1, 28*28])
  img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
  latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
  loss = tf.reduce_mean(img_loss + latent_loss)
  optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  
  print("\n============================================")
  print("                 Training Data                ")
  print("============================================\n")
  
  # Training

  for i in range(EPOCHS):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=BATCH_SIZE)[0]]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        
    if not i % 200:
      ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
      #plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
      #plt.show()
      #plt.imshow(d[0], cmap='gray')
      #plt.show()
      print("iter: " + str(i) + ", Loss: "+ str(ls) + ", mean: " + str(np.mean(i_ls)) + ", dist mean: " + str(np.mean(d_ls)) + ", mu: "+str(np.mean(mu)) + ", sigma: "+str(np.mean(sigm)))

  return sampled, dec, sess, keep_prob, np.mean(mu), np.mean(sigm), mnist, x_dim

def testModel():
  # load decoder, encoder, sampled and probability
  sampled, dec, sess, keep_prob, mu, sigma, mnist, x_dim = trainModel()
  
  print("\n============================================")
  print("                  Prediction                  ")
  print("============================================\n")
  # select a number
  while(1):
  
    number = str(input("Write a number sequence to predict: "))
    
    if number == "owari" :
      print("\n\nEnd Test\n")
      break
    
    if len(number) == 0:
      print("\nWrite number sequence !!\n")
      continue
    else:
      # generate captcha from numbers
      input_imgs = getListOfNumbers(mu, sigma, sess, x_dim, mnist, number, N_LATENT, sampled, dec, keep_prob)
      print("\nGenerated captcha ... \n")
      '''
      randoms = np.array([np.random.normal(0, 1, N_LATENT) for _ in range(1)])
      imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
      imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
      for img in imgs:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.show()
      '''
      generate_captcha(input_imgs)
