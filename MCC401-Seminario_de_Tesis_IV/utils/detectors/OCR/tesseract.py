#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract

# Optical Character Recognition
def OCR(image):
  text = pytesseract.image_to_string(Image.open(image))
  return text
