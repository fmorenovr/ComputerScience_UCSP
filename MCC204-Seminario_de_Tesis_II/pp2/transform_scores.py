#!/usr/bin/python

# -*- coding: utf-8 -*-

import os
import pandas
import numpy
import matplotlib.pyplot as plt
import zipfile
#import geocoder
from unicodedata import normalize

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# verifi dir
def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Read placepulse_2.zip
def read_zip_file(filepath):
  data = zipfile.ZipFile(filepath)
  return data

def preprocess_scores():
  namefile = "RioDeJaneiro.csv"
  summary = pandas.read_csv(namefile, names=['Position', 'Qscore'])
#  summary = pandas.read_csv(namefile)
  print(summary)
  Lat = [] #summary["Lat"]
  Lon = [] #summary["Lon"]
  Qscores = summary["Qscore"]

  for pos in summary["Position"]:
    str_pos = str.split(pos, "_")
    lat = str_pos[0]
    lon = str_pos[1]
    print(lat, lon)
    Lat.append(lat)
    Lon.append(lon)

  data = {"Lat": Lat,
          "Lon": Lon,
          "Qscore": Qscores,
  }

  aggregate_statistics = pandas.DataFrame(data, columns= ['Lat', 'Lon', "Qscore"])

  aggregate_statistics.to_csv(namefile, index=False)
  
  return


if __name__ == '__main__':
  preprocess_scores()
