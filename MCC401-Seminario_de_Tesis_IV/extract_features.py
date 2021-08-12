#!/usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

from utils.detectors import getGarbageDetections, getGarbageDetector
from utils.detectors import getGraffitiDetections, getGraffitiDetector

images_to_test = 'data/images/pp1/2011/'

model_graph = getGarbageDetector()
getGarbageDetections(images_to_test, model_graph, output_dir = 'outputs/features/garbage/')

model_graph = getGraffitiDetector()
getGraffitiDetections(images_to_test, model_graph, output_dir = 'outputs/features/graffiti/')

from utils.sceneparsers import getSceneParser, getSceneComponents

model_parser, gpu_id = getSceneParser()

_ = getSceneComponents(images_to_test, model_parser, gpu_id, sceneparser_dir="outputs/festures/sceneparser/")
    
