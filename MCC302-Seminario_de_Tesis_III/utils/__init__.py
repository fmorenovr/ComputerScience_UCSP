#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import os, glob

def verifyFile(files_list):
  return os.path.isfile(files_list)

def verifyType(file_name):
  if os.path.isdir(file_name):
    return "dir"
  elif os.path.isfile(file_name):
    return "file"
  else:
    return None

def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def load_images_path(images_path):
  """
  If image path is given, return it directly
  For txt file, read it and return each line as image path
  In other case, it's a folder, return a list with names of each
  jpg, jpeg and png file
  """
  input_path_extension = images_path.split('.')[-1]
  if input_path_extension in ['jpg', 'jpeg', 'png']:
    return [images_path]
  elif input_path_extension == "txt":
    with open(images_path, "r") as f:
      return f.read().splitlines()
  else:
    return glob.glob(
      os.path.join(images_path, "*.jpg")) + \
      glob.glob(os.path.join(images_path, "*.png")) + \
      glob.glob(os.path.join(images_path, "*.jpeg"))
