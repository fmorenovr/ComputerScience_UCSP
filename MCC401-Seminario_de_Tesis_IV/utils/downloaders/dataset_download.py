#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import os
import wget
import zipfile
from zipfile import ZipFile
import shutil

from utils import verifyDir, verifyFile

root_scores_dir="data/scores/"

def streetscore():
  url_to_download = "http://streetscore.media.mit.edu/static/files/streetscore_data.zip"
  dir_to_save = root_scores_dir+"streetscore/"
  verifyDir(dir_to_save)
  if not verifyFile(dir_to_save+"streetscore.zip"):
    if not verifyFile(dir_to_save+"streetscore_data.zip"):
      try:
        print("Downloading ...")
        wget.download(url_to_download, dir_to_save)
      except:
        print("streetscore", "File or dataset doesnt exist")
        return
    else:
      with zipfile.ZipFile(dir_to_save+"streetscore_data.zip", 'r') as zip_ref:
        zip_ref.extractall(dir_to_save)
      
      dirs = [f for f in os.listdir(dir_to_save) if os.path.isdir(dir_to_save+f)]
      
      os.chdir(dir_to_save)
      root_cwd = os.getcwd()
      
      os.chdir(dirs[0])
      temp_cwd = os.getcwd()
      files_in_dir = os.listdir(temp_cwd)
      
      zipObj = ZipFile('streetscore.zip', 'w')
      for file_name in files_in_dir:
        zipObj.write(file_name)
      zipObj.close()
      
      shutil.move(temp_cwd+"/streetscore.zip", root_cwd+"/streetscore.zip")
      os.chdir(root_cwd)
      try:
        os.rmdir(temp_cwd)
      except:
        shutil.rmtree(temp_cwd)
      print("Done")
  else:
    print("File Already exist")
  
def placepulse_1():
  url_to_download = "http://pulse.media.mit.edu/static/data/consolidated_data_jsonformatted.json"
  dir_to_save = root_scores_dir+"placepulse/pp1/"
  verifyDir(dir_to_save)
  if not verifyFile(dir_to_save+"placepulse_1.json"):
    if not verifyFile(dir_to_save+"consolidated_data_jsonformatted.json"):
      try:
        print("Downloading ...")
        wget.download(url_to_download, dir_to_save)
      except:
        print("placepulse", "File or dataset doesnt exist")
        return
    else:
      shutil.move(dir_to_save+"consolidated_data_jsonformatted.json", dir_to_save+"placepulse_1.json")
      print("Done")
  else:
    print("File Already exist")

def placepulse_2():
  url_to_download = "http://pulse.media.mit.edu/static/data/pp2_20161010.zip"
  dir_to_save = root_scores_dir+"placepulse/pp2/"
  verifyDir(dir_to_save)
  if not verifyFile(dir_to_save+"placepulse_2.zip"):
    if not verifyFile(dir_to_save+"pp2_20161010.zip"):
      try:
        print("Downloading ...")
        wget.download(url_to_download, dir_to_save)
      except:
        print("placepulse", "File or dataset doesnt exist")
        return
    else:
      with zipfile.ZipFile(dir_to_save+"pp2_20161010.zip", 'r') as zip_ref:
        zip_ref.extractall(dir_to_save)
      
      os.chdir(dir_to_save)
      
      os.rename("votes.csv", "placepulse_2.csv")
      
      zipObj = ZipFile('placepulse_2.zip', 'w')
      zipObj.write("placepulse_2.csv")
      zipObj.close()
      
      os.remove("readme")
      os.remove("pp2_20161010.zip")
      
      try:
        os.rmdir("__MACOSX/")
      except:
        shutil.rmtree("__MACOSX/")
      print("Done")
  else:
    print("File Already exist")

def download_dataset(dataset="placepulse_1"):
  if dataset=="streetscore":
    streetscore()
  elif dataset=="placepulse_1":
    placepulse_1()
  elif dataset=="placepulse_2":
    placepulse_2()
  else:
    streetscore()
    placepulse_1()
    placepulse_2()
