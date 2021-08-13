import numpy as np
import pandas as pd
import os

import six.moves.urllib as urllib
import tensorflow as tf
import time
import glob

from tqdm import tqdm
from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .utils import visualization_utils as vis_util
from .utils import label_map_util

from utils import verifyDir, load_images_path

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/detectors/mask_rcnn/garbage/frozen_inference_graph.pb'
PATH_TO_LABELS = 'utils/detectors/garbage/annotations/label_map.pbtxt'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_garbage(image_path, parameters, output_dir="", max_boxes=10, min_thres=0.59, save_labels=True, dont_show=False):
    
  [sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections] = parameters
  
  image = Image.open(image_path)
  im_width, im_height = image.size
  file_name = (image_path.split("/")[-1]).split(".")[0]
  image_np = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image_np, axis=0)

  (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
  
  d_class={}
  if save_labels:
    d_class={}
    d_class["ID"] = file_name
    for k in categories:
      d_class[k['name']] = [0]
    
    with open(output_dir+file_name+".csv", "w") as f:
      f.write("class,x,y,w,h,confidence,area\n")
      for i in range(max_boxes):
        ymin, xmin, ymax, xmax = boxes[0][i]
        top, left, bottom, right = ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width
        label = categories[int(classes[0][i])-1]['name']
        confidence = scores[0][i]
        area = (right-left)* (bottom-top)/(im_width*im_height)
        if confidence > min_thres:
          d_class[label][0] = d_class[label][0]+1
          f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(label, left, top, right-left, bottom-top, float(confidence),area*100))
    f.close()
    
  if not dont_show:
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
        np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), CATEGORY_INDEX,
        min_score_thresh=min_thres, use_normalized_coordinates=True, line_thickness=4, max_boxes_to_draw=max_boxes)

    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image_np, aspect = 'auto')
    plt.savefig(output_dir+'{}'.format(file_name+'_detections.jpg'), dpi = 62)
    plt.close(fig)
    
    import cv2
    cv2_img = cv2.imread(output_dir+file_name+"_detections.jpg")
    cv2_img = cv2.resize(cv2_img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_dir+file_name+"_detections.jpg", cv2_img)
  
  return pd.DataFrame(data=d_class)

# Load model into memory
def getGarbageDetector(model_path=PATH_TO_CKPT):
  print('Loading model...')
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def getGarbageDetections(input_args, detection_graph, output_dir='', max_boxes=10, min_thres=0.59, save_labels=True, dont_show=False):
  print('***************************')
  print('**** GARBAGE GETECTION ****')
  print('***************************')
  
  verifyDir(output_dir)
  
  images_to_test = load_images_path(input_args)
  
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
      parameters = [sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections]
      
      pbar = tqdm(total=len(images_to_test))
      data_objects = pd.DataFrame()
      for image_path in images_to_test:
        detections_obj = detect_garbage(image_path, parameters, output_dir, max_boxes, min_thres, save_labels, dont_show)
        data_objects = data_objects.append(detections_obj, ignore_index=True)
        print('***********************')
        pbar.update(1)
      data_objects.to_csv(output_dir+'objects_detected.csv', index=False)

if __name__ == '__main__':
  images_to_test = 'data/images/pp1/2011/'

  model_graph = getGarbageDetector()
  getGarbageDetections(images_to_test, model_graph, output_dir = 'outputs/')
