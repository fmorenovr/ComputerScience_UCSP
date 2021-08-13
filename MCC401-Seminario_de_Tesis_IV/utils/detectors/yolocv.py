#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

import numpy as np
import time
import cv2

import time
#from . import darknet as dn
from utils.datasets import getImageName

YOLO_dir = "data/categories/"
YOLO_MODEL_DIR = "data/outputs/detection/"

def getYOLOModel(model_name="yolov3", dataset="coco"):
  cfg = "utils/detectors/config/"+model_name+"-"+dataset+".cfg"
  weights = "models/detectors/"+model_name+"/"+model_name+"-"+dataset+".weights"
  data = YOLO_dir+dataset+"/"+dataset+".data"
  names = YOLO_dir+dataset+"/"+dataset+".names"
  return cfg, weights, data, names

'''
def YOLO(image, infoNet, LABELS, detection_dir="", confThreshold = 0.4, nmsThreshold = 0.6):
  #dn.set_gpu(0)
  img_name = getImageName(image)
  infoImg = [image, img_name+"_detections"]
  print("[INFO] loading YOLO from disk ...")
  start = time.time()
  results = dn.detect(infoImg, infoNet, detection_dir, confThreshold, nmsThreshold)
  print(results)
  end = time.time()
  print("[INFO] YOLO took {:.6f} seconds".format(end - start))
  print("[INFO] Number of Objects Detected: "+str(len(results)))
  for clss, prob, _ in results:
    print("- {}: {:.4f}".format(clss, prob))
  return results

def Yolo(cfg, weights, data, names):
  LABELS = open(names).read().strip().split("\n")
  net = dn.load_net(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
  meta = dn.load_meta(data.encode('utf-8'))
  return [net, meta], LABELS
'''

def drawBoxes(img, results):
  image = cv2.imread(img)
  for result in results:
    drawBox(image, result[0], result[1], result[2], result[3])
  return image

# Draw the predicted bounding box
def drawBox(frame, classPred, conf, bbox, color, justname=True):
  x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
  # load our input image and grab its spatial dimensions
  # Draw a bounding box.
  cv2.rectangle(frame, (x, y), (w, h), color, 2)
  
  if justname:
    text = "{}".format(classPred)
  else:
    text = "{}: {:.4f}".format(classPred, conf)
      
  #Display the label at the top of the bounding box
  labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
  top = max(y, labelSize[1])
  cv2.rectangle(frame, (x, y - round(1.5*labelSize[1])), (x + round(1.5*labelSize[0]), y + baseLine), color, cv2.FILLED)
  cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Get the names of the output layers
def getOutputsNames(net):
  # Get the names of all the layers in the network
  layersNames = net.getLayerNames()
  # Get the names of the output layers, i.e. the layers with unconnected outputs
  return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, classes, layerOutputs, confThreshold = 0.5, nmsThreshold = 0.3):
  H, W = frame.shape[0], frame.shape[1]
  # initialize a list of colors to represent each possible class label
  np.random.seed(42)
  COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
  # Scan through all the bounding boxes output from the network and keep only the
  # ones with high confidence scores. Assign the box's class label as the class with the highest score.
  classIds = []
  confidences = []
  boxes = []
  # loop over each of the layer outputs
  for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability) of
      # the current object detection
      scores = detection[5:]
      classId = np.argmax(scores)
      confidence = scores[classId]
      
      # filter out weak predictions by ensuring the detected
      # probability is greater than the minimum probability
      if confidence > confThreshold:
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        # update our list of bounding box coordinates, confidences,
        # and class IDs
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIds.append(classId)

  # Perform non maximum suppression to eliminate redundant overlapping boxes with
  # lower confidences.
  indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
  
  print("[INFO] Number of Objects Detected: "+str(len(indices)))
  
  results=[]
  
  if len(indices) > 0:
    for i in indices.flatten():
      # extract the bounding box coordinates
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      color = [int(c) for c in COLORS[classIds[i]]]
      conf = confidences[i]
      classPred = classes[classIds[i]]
      print("- {}: {:.4f}".format(classPred, conf))
      results.append([classPred, conf, [x, y ,x+w ,y+h], color])
  return results

def cvYolo(cfg, weights, data, names):
  LABELS = open(names).read().strip().split("\n")
  net = cv2.dnn.readNetFromDarknet(cfg, weights)
  return net, LABELS

def cvYOLO(img, net, LABELS, detection_dir="", confThreshold = 0.4, nmsThreshold = 0.6):

  # load our input image and grab its spatial dimensions
  image = cv2.imread(img)

  # determine only the *output* layer names that we need from YOLO
  ln = getOutputsNames(net)

  # construct a blob from the input image and then perform a forward
  # pass of the YOLO object detector, giving us our bounding boxes and
  # associated probabilities
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  print("[INFO] loading YOLO from disk ...")
  start = time.time()
  layerOutputs = net.forward(ln)
  end = time.time()
  print("[INFO] YOLO took {:.6f} seconds".format(end - start))

  detections = postprocess(image, LABELS, layerOutputs, confThreshold,nmsThreshold)
  #t, _ = net.getPerfProfile()
  #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
  #cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
  #drawBox(frame, classes[classIds[i]], confidences[i], x, y, x + w, y + h, color)
  if detection_dir=="":
    return detections
  else:
    img_name = getImageName(img)
    img_objs = drawBoxes(img, detections)
    cv2.imwrite(detection_dir+img_name+"_detections.png", img_objs)

  return detections

