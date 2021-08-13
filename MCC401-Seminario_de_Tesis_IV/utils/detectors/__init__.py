#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import collections
import cv2


from .yolocv import cvYOLO, getYOLOModel, cvYolo#, YOLO, Yolo
from utils import verifyDir

from .garbage import getGarbageDetections, getGarbageDetector
from .graffiti import getGraffitiDetections, getGraffitiDetector
from .OCR import getTextsDetections

def getObjectDetector(model_name="yolov3", dataset="coco", type_dect="cvyolo"):
  if model_name == "yolov3":
    net, LABELS = getYOLODetector(model_name, dataset, type_dect)
  else:
    net, LABELS = getYOLODetector(model_name, dataset, type_dect)
  return net, LABELS

def getYOLODetector(model_name="yolov3", dataset="coco", type_dect="cvyolo"):
  cfg, weights, data, names = getYOLOModel(model_name, dataset)
  if type_dect == "cvyolo":
    net, LABELS = cvYolo(cfg, weights, data, names)
  #elif type_dect == "darknet":
  #  net, LABELS = DarkNetYolo(cfg, weights, data, names)
  else:
    net, LABELS = cvYolo(cfg, weights, data, names)
  return net, LABELS

def getYOLODetections(image, net, LABELS, detection_dir="", type_dect="cvyolo", confThreshold = 0.4, nmsThreshold = 0.6):
  if type_dect == "cvyolo":
    detections = cvYOLO(image, net, LABELS, detection_dir, confThreshold, nmsThreshold)
  #elif type_dect == "darknet":
  #  detections = YOLO(image, net, LABELS, detection_dir, confThreshold, nmsThreshold)
  else:
    detections = cvYOLO(image, net, LABELS, detection_dir, confThreshold, nmsThreshold)
  return detections

def getObjectsDetections(image, net, LABELS, detection_dir="", model_name="yolov3", dataset="coco", type_dect="cvyolo", confThreshold = 0.4, nmsThreshold = 0.6):
  print('--------------------------')
  print('---- OBJECT DETECTION ----')
  print('--------------------------')
  if detection_dir != "":
    verifyDir(detection_dir)
  if model_name == "yolov3":
    detections = getYOLODetections(image, net, LABELS, detection_dir, type_dect, confThreshold, nmsThreshold)
  else:
    detections = getYOLODetections(image, net, LABELS, detection_dir, type_dect, confThreshold, nmsThreshold)
  print('--------------------------')
  return detections

