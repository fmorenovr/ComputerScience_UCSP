#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from .tesseract import OCR

def getTextsDetections(image, type_dect="ocr"):
  if type_dect == "ocr":
    detections = OCR(image)
  else:
    detections = OCR(image)
  return detections
