# System libs
import os
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv

from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat

# Our libs

from .lib.nn import user_scattered_collate, async_copy_to
from .lib.config import cfg
from .lib.utils import as_numpy, colorEncode, find_recursive, setup_logger, TestDataset
from .lib.networks import ModelBuilder, SegmentationModule
'''
from lib.nn import user_scattered_collate, async_copy_to
from lib.config import cfg
from lib.utils import as_numpy, colorEncode, find_recursive, setup_logger, TestDataset
from lib.networks import ModelBuilder, SegmentationModule
'''

def getComponentsCategories(colors_path, names_path):
  colors = loadmat(colors_path)['colors']
  names = {}
  with open(names_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
      names[int(row[0])] = row[5].split(";")[0]
  return names, colors

def visualize_result(data, pred, cfg, dont_show=False):
  (img, info) = data

  img_name = info.split('/')[-1]

  # print predictions in descending order
  pred = np.int32(pred)
  pixs = pred.size
  uniques, counts = np.unique(pred, return_counts=True)
  
  scene_components = []
  
  names, colors = getComponentsCategories(cfg.CATEGORIES.colors, cfg.CATEGORIES.names)

  file_name = img_name.split(".")[0]
  d_class={}
  d_class["ID"] = file_name
  for k,v in names.items():
    d_class[v] = [0]

  print("[INFO] Predictions in [{}]".format(info))
  with open(cfg.TEST.result+file_name+".csv", "w") as f:
    f.write("class,ratio\n")
    for idx in np.argsort(counts)[::-1]:
      name = names[uniques[idx] + 1]
      color = colors[uniques[idx]]
      ratio = counts[idx] / pixs * 100
      if ratio > 0.1:
        print("{} - {}: {:.2f}%".format(color, name, ratio))
        scene_components.append([color, name, ratio])
      f.write("{},{:.4f}\n".format(name, float(ratio)))
      d_class[name][0] = ratio
    f.close()
  
  if not dont_show:
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    #img_result = pred_color*0.6 + img # need to add opacity to pred_color and then overlap it on img
    #Image.fromarray(img_result.astype(np.uint8)).save("/home/jenazads/github/scene_parsing.png")

    Image.fromarray(pred_color).save(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '_scene_parsing.png')))
    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    Image.fromarray(im_vis).save(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))
  
  return pd.DataFrame(data=d_class), scene_components

def getSceneComponents(img, segmentation_module, gpu_id=0, sceneparser_dir="", save_labels=True, dont_show=False):
  loader = getScenePreprocess(img, sceneparser_dir)

  print('***********************')
  print('**** SCENE PARSING ****')
  print('***********************')
  segmentation_module.cuda()
  segmentation_module.eval()
  
  data_objects = pd.DataFrame()
  pbar = tqdm(total=len(loader))
  for batch_data in loader:
    # process data
    batch_data = batch_data[0]
    segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
      scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
      scores = async_copy_to(scores, gpu_id)

      for img in img_resized_list:
        feed_dict = batch_data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        del feed_dict['info']
        feed_dict = async_copy_to(feed_dict, gpu_id)

        # forward pass
        pred_tmp = segmentation_module(feed_dict, segSize=segSize)
        scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

      _, pred = torch.max(scores, dim=1)
      pred = as_numpy(pred.squeeze(0).cpu())

    # visualization
    
    scene_components_df, scene_components = visualize_result((batch_data['img_ori'], batch_data['info']), pred, cfg)
    if save_labels:
      data_objects = data_objects.append(scene_components_df, ignore_index=True)
    print('***********************')
    pbar.update(1)
  
  data_objects.to_csv(cfg.TEST.result+'objects_segmented.csv', index=False)
  
  return scene_components

def getSceneParser(cfg_file=None, gpu_id=0):
  if cfg_file==None:
    cfg_file = "utils/sceneparsers/lib/config/files/ade20k-resnet50dilated-ppm_deepsup.yaml"
  cfg = setSceneParserConfig(cfg_file)

  torch.cuda.set_device(gpu_id)
  # Network Builders
  net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=os.path.join(cfg.DIR,cfg.MODEL.weights_encoder))

  net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=os.path.join(cfg.DIR, cfg.MODEL.weights_decoder),
    use_softmax=True)

  crit = nn.NLLLoss(ignore_index=-1)

  segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

  return segmentation_module, gpu_id

def setSceneParserConfig(cfg_file, do_logger=False):

  cfg.merge_from_file(cfg_file)

  if do_logger:
    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(cfg_file))
    logger.info("Running with config:\n{}".format(cfg))
  return cfg

def getScenePreprocess(img, sceneparser_dir=""):

  # generate testing image list
  if os.path.isdir(img):
    imgs = find_recursive(img)
  else:
    imgs = [img]
  assert len(imgs), "imgs should be a path to image (.jpg) or directory."
  cfg.list_test = [{'fpath_img': x} for x in imgs]
  
  if sceneparser_dir != "":
    cfg.TEST.result = sceneparser_dir

  if not os.path.isdir(cfg.TEST.result):
    os.makedirs(cfg.TEST.result)

  # Dataset and Loader
  dataset_test = TestDataset(
    cfg.list_test,
    cfg.DATASET)
  
  loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=cfg.TEST.batch_size,
    shuffle=False,
    collate_fn=user_scattered_collate,
    num_workers=5,
    drop_last=True)
  
  return loader_test

if __name__ == '__main__':
  assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

  cfg_file = "utils/sceneparsers/lib/config/files/ade20k-resnet50dilated-ppm_deepsup.yaml"

  cfg = setSceneParserConfig(cfg_file)

  segmentation_module, gpu_id = getSceneParser(cfg, 0)

  scene_components = getSceneComponents("178.jpg", segmentation_module, gpu_id)

  print('Inference done!')
