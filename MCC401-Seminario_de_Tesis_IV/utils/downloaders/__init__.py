#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from .image_download import is_valid_image, is_available_image, download_panorama
from .dataset_download import download_dataset

if __name__ == "__main__":
  API_key = ""
  pano_dir = "panoids/"
  download_panorama(-22.9361399,-43.1772293, dir_path=pano_dir, API_key=API_key)
  
  #40.75388056, -73.99697222
  #40.6932246,-73.9724456
