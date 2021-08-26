#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import cv2
import numpy as np
import pandas as pd

from utils import load_images_path

images_path = load_images_path('data/images/pp1/2011/')

garbage_dir = "outputs/detector/garbage/"
graffiti_dir = "outputs/detector/graffiti/"
objects_dir = "outputs/detector/objects/"

dirs_list = [
             [garbage_dir, "objects_detected.csv", "garbage_segmented.csv"],
             [objects_dir, "objects_detected.csv", "objects_segmented.csv"]
             [graffiti_dir, "objects_detected.csv", "graffiti_segmented.csv"]
             ]

for cur_dir, summary_file, output_file in dirs_list:
  summ_df = pd.read_csv(cur_dir+summary_file)
  output_df = pd.DataFrame()
  for img_name in images_path:
    file_name = (img_name.split("/")[-1]).split(".")[0]
    summ_file = summ_df[summ_df['ID']==int(file_name)]
    img_aux = cv2.imread(img_name)
    df = pd.read_csv(cur_dir+file_name+'.csv')
    height_, width_, _ = img_aux.shape # Y, X
    classes = df["class"].unique()
    for class_ in classes:
      img_white = np.zeros((height_, width_),dtype=np.uint8)
      img_white.fill(255)
      for obj in df[df["class"]==class_].itertuples():
        x, y, w, h = obj[2], obj[3], obj[4], obj[5]
        
        max_x = round(x+w) if round(x+w)<=width_ else width_
        max_y = round(y+h) if round(y+h)<=height_ else height_
        
        cv2.rectangle(img_white, (round(x), round(y)), (max_x, max_y), (0), -1)
      
      cntNotBlack = cv2.countNonZero(img_white)
      height, width = img_white.shape
      cntPixels = height*width
      cntBlack = cntPixels - cntNotBlack
      
      summ_file[class_] = float(cntBlack/cntPixels)
      '''
      print(x, y, w, h, w*h, w*h/cntPixels)
      print(round(x), round(y), round(x+w), round(y+h))
      print(int(x), int(y), int(x+w)-int(x)+1, int(y+h)-int(y)+1, (int(x+w)-int(x)+1)*(int(y+h)-int(y)+1), (int(x+w)-int(x)+1)*(int(y+h)-int(y)+1)/cntPixels)
      print(cntPixels, cntBlack, float(cntBlack/cntPixels))
      '''
      cv2.imwrite(cur_dir+file_name+'_rects_'+class_+'.png',img_white)
      
    output_df = output_df.append(summ_file, ignore_index=True)
  output_df.to_csv(cur_dir+output_file, index=False)

