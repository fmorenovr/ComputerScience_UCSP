#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import urllib.request
import json
from PIL import Image
import os
import cv2
import numpy

from . import streetview

from utils import verifyDir

#https://maps.googleapis.com/maps/api/streetview?size=640x420&heading=180&key=AIzaSyBSZXAK_Gc7S8O_Go_lNMOXF0PMtcQ-yB4&fov=90&location=40.75388056, -73.99697222

#https://maps.googleapis.com/maps/api/streetview?size=640x420&heading=180&key=AIzaSyBSZXAK_Gc7S8O_Go_lNMOXF0PMtcQ-yB4&fov=90&pano=EF4rAqoXGeQi3EXZRchxJg

image_endpoint = 'https://maps.googleapis.com/maps/api/streetview'
indexToYMapping = {0: 0, 1: 15, 2: 35, 3: 20}
API_key = ""

wrong_image_path = 'utils/downloaders/none_image.jpg'
wrong_image = cv2.imread(wrong_image_path)

blank_image_path = 'utils/downloaders/blank_image.jpg'
blank_image = cv2.imread(blank_image_path)

def is_valid_image(image_path):
  image_aux = cv2.imread(image_path)
  image_aux = cv2.resize(image_aux, dsize=(640, 420), interpolation=cv2.INTER_LINEAR)
  dist_wrong = numpy.linalg.norm((image_aux - wrong_image).flatten(), ord=2)
  
  dist_blank = numpy.linalg.norm((image_aux - blank_image).flatten(), ord=2)
  
  return dist_wrong > 0.001 or dist_blank > 0.001

def is_available_image(lat, lon, API_key = API_key):
  url = image_endpoint + f'/metadata?location={lat},{lon}&key={API_key}'
  print("Verifying url: ", url)
  response = urllib.request.urlopen(url).read()
  image_available = json.loads(response).get('status') == 'OK'
  return image_available

def download_panoramic(lat, lon, size_img=[640, 420], API_key = API_key, dir_path="./"):
  img_name = str(lat)+"_"+str(lon)
  image_available = is_available_image(lat, lon, API_key)
  print("Lat: ", lat, " Lon: ", lon, "API_key: ", API_key)
  if not image_available:
    print(f'image not available at {lat}, {lon}')
    return False

  canvas = Image.new('RGB', (4 * size_img[0], size_img[1]))
  for heading in range(0, 360, 90):
    print(f'Fetching {lat},{lon} -- heading {heading}')
    url = image_endpoint + f'?key={API_key}&fov=90&size={size_img[0]}x{size_img[1]}&heading={heading}&location={lat},{lon}'
    img =  Image.open(urllib.request.urlopen(url))
    x = int((heading / 90) * 640)
    y = indexToYMapping.get(int(heading/90))
    canvas.paste(img, (x, y))
  maxDepression = max(list(indexToYMapping.values()))
  width, height = canvas.size
  canvas.crop((0, maxDepression, width, height)).save(dir_path+img_name+".png")
  return True

def download_panoids(lat, lon, size_img=[640, 420], API_key = API_key, dir_path="./"):
  image_available = is_available_image(lat, lon, API_key)
  print("Lat: ", lat, " Lon: ", lon, "API_key: ", API_key)
  if not image_available:
    print(f'image not available at {lat}, {lon}')
    return False
  
  panoids = streetview.panoids(lat=lat, lon=lon)
  print(panoids)
  for panoid in panoids:
    img_name = str(lat)+"_"+str(lon)
    '''
    try:
      if panoid['month']:
        img_name = str(panoid['month'])+"_"+img_name
    except:
      img_name = img_name
    
    try:
      if panoid['year']:
        img_name = str(panoid['year'])+"_"+img_name
    except:
      img_name = img_name
    '''
    try:
      if panoid['year'] and panoid['month']:
        img_name = str(panoid['year'])+"_"+str(panoid['month'])+"_"+img_name
    except:
      continue

    pano_id = panoid['panoid']
    #img_name = pano_id+"_"+img_name
    
    panorama = streetview.download_panorama_v3(panoid['panoid'], zoom=2, disp=False)
    cv2.imwrite(dir_path+"Pano_"+img_name+".png", cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
   
    canvas = Image.new('RGB', (4 * size_img[0], size_img[1]))
    for heading in range(0, 360, 90):
      print(f'Fetching {lat},{lon} -- heading {heading} -- panoid {pano_id}')
      url = image_endpoint + f'?key={API_key}&fov=90&size={size_img[0]}x{size_img[1]}&heading={heading}&pano={pano_id}'
      img =  Image.open(urllib.request.urlopen(url))
      x = int((heading / 90) * 640)
      y = indexToYMapping.get(int(heading/90))
      canvas.paste(img, (x, y))
    maxDepression = max(list(indexToYMapping.values()))
    width, height = canvas.size
    canvas.crop((0, maxDepression, width, height)).save(dir_path+img_name+".png")
  return True
   
def download_panorama(lat, lon, pano_id=True, dir_path="./", size_img=[640, 420], API_key = API_key):
  verifyDir(dir_path)
  if pano_id:
    download_panoids(lat, lon, size_img, API_key, dir_path)
  else:
    download_panoramic(lat, lon, size_img, API_key, dir_path)

if __name__ == "__main__":
  API_key = ""
  pano_dir = "panoids/"

  download_panorama(40.75388056, -73.99697222, dir_path=pano_dir, API_key=API_key)
