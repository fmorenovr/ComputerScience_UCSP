#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def mse(imageA, imageB):
  # the 'Mean Squared Error' between the two images is the
  # sum of the squared difference between the two images;
  # NOTE: the two images must have the same dimension
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1])

  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err

def compare_images(imageA, imageB, title):
  # compute the mean squared error and structural similarity
  # index for the images
  m = mse(imageA, imageB)
  s = ssim(imageA, imageB)
  # setup the figure
  fig = plt.figure(title)
  plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
  # show first image
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(imageA, cmap = plt.cm.gray)
  plt.axis("off")
  # show the second image
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(imageB, cmap = plt.cm.gray)
  plt.axis("off")
  # show the images
  plt.show()

# Function to fill all the bounding box
def fill_rects(image, stats):
  for i,stat in enumerate(stats):
    if i > 0:
      p1 = (stat[0],stat[1])
      p2 = (stat[0] + stat[2],stat[1] + stat[3])
      cv2.rectangle(image,p1,p2,255,-1)


# Load image file
img1 = cv2.resize(cv2.imread('test/img_1.png',0), (800, 600))
img2 = cv2.resize(cv2.imread('test/img_2.png',0), (800, 600))

print(img1.shape)
print(img2.shape)

# Subtract the 2 image to get the difference region
img3 = cv2.subtract(img1,img2)

# Make it smaller to speed up everything and easier to cluster
small_img = cv2.resize(img3,(0,0),fx = 0.25, fy = 0.25)

# Morphological close process to cluster nearby objects
fat_img = cv2.dilate(small_img, None,iterations = 3)
fat_img = cv2.erode(fat_img, None,iterations = 3)

fat_img = cv2.dilate(fat_img, None,iterations = 3)
fat_img = cv2.erode(fat_img, None,iterations = 3)

# Threshold strong signals
_, bin_img = cv2.threshold(fat_img,20,255,cv2.THRESH_BINARY)

# Analyse connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

# Cluster all the intersected bounding box together
rsmall, csmall = np.shape(small_img)
new_img1 = np.zeros((rsmall, csmall), dtype=np.uint8)

fill_rects(new_img1,stats)


# Analyse New connected components to get final regions
num_labels_new, labels_new, stats_new, centroids_new = cv2.connectedComponentsWithStats(new_img1)


labels_disp = np.uint8(200*labels/np.max(labels)) + 50
labels_disp2 = np.uint8(200*labels_new/np.max(labels_new)) + 50

compare_images(img1, img2, "Original vs. Original")

cv2.imshow('diff',img3)
cv2.imshow('small_img',small_img)
cv2.imshow('fat_img',fat_img)
cv2.imshow('bin_img',bin_img)
cv2.imshow("labels",labels_disp)
cv2.imshow("labels_disp2",labels_disp2)
cv2.waitKey(0)
