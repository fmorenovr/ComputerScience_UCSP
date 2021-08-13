#!/bin/bash

# Inference
python3 -u test.py --imgs $1 --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
# DIR $MODEL_PATH TEST.result ./ TEST.checkpoint epoch_20.pth
