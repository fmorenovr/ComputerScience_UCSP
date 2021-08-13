"""
Mask R-CNN
Train on the toy Graffiti dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 graffiti.py train --dataset=/path/to/graffiti/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 graffiti.py train --dataset=/path/to/graffiti/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 graffiti.py train --dataset=/path/to/graffiti/dataset --weights=imagenet

    # Apply color splash to an image
    python3 graffiti.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 graffiti.py splash --weights=last --video=<URL or path to file>
    
    # To run
    python3 graffiti.py batchimg --weights models/mask_rcnn_graffiti_0029.h5 --imdir tests --outdir outputs
"""

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import random
import imgaug
import imageio
import shutil
from utils import verifyDir, load_images_path
import matplotlib.pyplot as plt


ROOT_DIR = "models/detectors/mask_rcnn/graffiti/"

#sys.path.append(ROOT_DIR)  # To find local version of the library
from .mrcnn import visualize
from .mrcnn.config import Config
from .mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

_CLASSES = [ "tag", "frame"]
NUM_CLASSES = 1 + len(_CLASSES)  # Background
############################################################
#  Configurations
############################################################


class GraffitiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

#class MyConfig(Config):

    # Setting other parameters...

    def __init__(self, nclasses):
        self.NUM_CLASSES = nclasses
        super().__init__()

    # Give the configuration a recognizable name
    NAME = "graffiti"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    DETECTION_NMS_THRESHOLD = 0.3
    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.7
    LEARNING_RATE = 0.0007
    LEARNING_MOMENTUM = 0.7

class InferenceConfig(GraffitiConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Dataset
############################################################

class GraffitiDataset(utils.Dataset):

    def load_graffiti(self, dataset_dir, subset):
        """Load a subset of the Graffiti dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        for j, cl in enumerate(_CLASSES, 1):
            self.add_class("graffiti", j, cl)
        #self.add_class("graffiti", 1, "tag")
        #self.add_class("graffiti", 2, "frame")
        #self.add_class("graffiti", 3, "sign")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            classes = [r['region_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "graffiti",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a graffiti dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "graffiti":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        classes = np.array([ x['class'] for x in info['classes'] ])
        classesids = classes.copy()
        for j, cl in enumerate(_CLASSES, 1):
            classesids[classes == cl] = j
        return mask.astype(np.bool), classesids.astype(int)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "graffiti":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, model_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = GraffitiDataset()
    dataset_train.load_graffiti(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GraffitiDataset()
    dataset_val.load_graffiti(args.dataset, "val")
    dataset_val.prepare()

    mAP_freq = 5

    class ValConfig(GraffitiConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.0

    #myconfig = ValConfig(nclasses=NUM_CLASSES)
    model_inference = modellib.MaskRCNN(mode="inference",
                                        config=ValConfig(NUM_CLASSES),
                                        model_dir=model_dir)

    mAP_callback = modellib.MeanAveragePrecisionCallback(model,
                                                         model_inference,
                                                         dataset_val,
                                                         mAP_freq, verbose=1)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                #layers='heads')
                layers='heads',
                augmentation=imgaug.augmenters.Sequential([
                    imgaug.augmenters.Crop(px=(0, 50)), # crop images from each side by 0 to 16px (randomly chosen)
                    imgaug.augmenters.Fliplr(0.5), # horizontally flip 50% of the images
                    imgaug.augmenters.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
                #]))
                ]),
                custom_callbacks=[mAP_callback],)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    #print('##########################################################')
    #print(mask)
    if mask.size == 0: return gray
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #print(mask)
    # Copy color pixels from the original color image where mask is set
    #if mask.shape[0] > 0:
    #if mask.size > 0:
    splash = np.where(mask, image, gray).astype(np.uint8)
    #else:
        #splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, imdir=None, video_path=None, outdir='/tmp'):
    # Does NOT work well for more than one class
    #assert image_path or video_path

    # Image or video?
    if image_path:
        #print("Running on {}".format(image_path))
        image = skimage.io.imread(image_path)
        r = model.detect([image], verbose=0)[0]
        #print(r)
        splash = color_splash(image, r['masks'])
        outpath = os.path.basename(image_path)
        skimage.io.imsave(os.path.join(outdir, outpath), splash)
    elif imdir:
        filenames = os.listdir(imdir)
        imgs = []
        for f in filenames:
            if f.endswith('.jpg'):
                imgs.append(os.path.join(imdir, f))

        for img in imgs:
            print("Running on {}".format(img))
            image = imageio.imread(img)
            r = model.detect([image], verbose=0)[0]
            #print(r)
            splash = color_splash(image, r['masks'])
            outpath = os.path.basename(img)
            imageio.imwrite(os.path.join(outdir, outpath), splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def find_last_idx(_dir, fileslist, ext='.csv'):
    """Finds the maximum index of the files contained in @_dir,
    given the list given by fileslist

    Args:
    _dir(str): dir path
    fileslist(list of str): list of filenames without the extension

    Returns:
    integer: maximum index
    """
    dirfiles = sorted(os.listdir(_dir), reverse=True) # reverse order
    for f in dirfiles:
        if not f.endswith(ext): continue
        idx = fileslist.index(f.replace(ext, ''))
        return idx
    return 0

def batchcsv(model, imdir, outdir, reverse=False, shuffle=False):
    print('Batch mode')
    aux = sorted(os.listdir(imdir))

    if reverse:
        aux = reversed(aux)
    elif shuffle:
        aux = random.sample(aux, k=len(aux))

    imgfiles = []
    for f in aux:
        if f.endswith('.jpg'):
            imgfiles.append(f.replace('.jpg', ''))

    IMGSLISFILE = 'imgs.lst'
    listpath = os.path.join(outdir, IMGSLISFILE)

    if not os.path.exists(listpath):
        listfh = open(listpath, 'w')
        for item in imgfiles:
            listfh.write(item+ '\n')
        listfh.close()

    batchsz = model.config.BATCH_SIZE
    imgs = []
    outcsvs = []
    nimgs = 0

    #lastfileidx = 0
    lastfileidx = find_last_idx(outdir, imgfiles)
    print('Starting from file index:{}'.format(lastfileidx))

    for fileidx, filename in enumerate(imgfiles[lastfileidx:]):

        #filenameroot = filename.replace('.jpg', '')
        outcsv = os.path.join(outdir, filename + '.csv')
        if os.path.exists(outcsv): continue

        print('{}: {}'.format(fileidx + lastfileidx, filename))

        if nimgs == batchsz:
            rs = model.detect(imgs, verbose=0)

            for j, r in enumerate(rs):
                outcsv = outcsvs[j]
                bboxes = r['rois']
                finalstr = ''

                for i in range(len(bboxes)):
                    c = r['class_ids'][i]; b = bboxes[i]; s = r['scores'][i]
                    if c != 1: continue # we are just interested on graffiti tag
                    origarea = np.sum(r['masks'][:, :, i])
                    finalstr += '{},{},{},{},{},{},{}\n'. \
                        format(b[1],b[0], b[3], b[2], origarea, c, s)

                if finalstr != '':
                    with open(outcsv, 'w') as fh:
                        fh.write(finalstr)

            imgs = []
            outcsvs = []
            nimgs = 0

        imgs.append(skimage.io.imread(os.path.join(imdir, filename + '.jpg')))
        outcsvs.append(outcsv)
        nimgs += 1

def batch_img(model, imdir, outdir="", reverse=False, shuffle=False, min_thres=.59, exclude_class=['frame', 'sign'], save_labels=True, dont_show=False):
    print('Batch img mode')

    aux = load_images_path(imdir)
    temp_dir = outdir
    if reverse:
        files = reversed(aux)
    elif shuffle:
        files = random.sample(aux, k=len(aux))
    else:
        files = aux

    classes = ['bg', 'tag', 'frame', 'sign']
    #fig, ax = plt.subplots(1,1, figsize=(10, 10))
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    for filename in files:
      if filename.endswith('.png'):
        try:
            from PIL import Image
        except ImportError:
            import Image
        im = Image.open(filename)
        im = im.convert('RGB')
        im.save(os.path.join(temp_dir, filename.replace('.png', '.jpg')))
    
    pbar = tqdm(total=len(files))
    data_objects = pd.DataFrame()
    for filename in files:
      if not filename.endswith('.jpg'):
        outpath = os.path.join(temp_dir, filename.split(".")[0]+"jpg" )
        
        if not os.path.exists(outpath) or filename.endswith('.png'):
            continue
      plt.cla()

      img = imageio.imread(filename)
      
      r = model.detect([img], verbose=0)[0]

      masks, boxes, scores, class_ids = r['masks'], r['rois'], r['scores'], r['class_ids']
      im_width, im_height, N = masks.shape
      
      areas_mask = masks.sum(axis=(0,1))/(im_width*im_height)

      file_name = (filename.split("/")[-1]).split(".")[0]
      if save_labels:
        d_class={}
        d_class["ID"] = file_name
        d_class['graffiti']=[0]
        d_class['graffiti_mask']=[0]
        with open(outdir+file_name+".csv", "w") as f:
          label = 'graffiti'
          f.write("class,x,y,w,h,confidence,area_rect, area_mask\n")
          for i in range(N):
            if classes[class_ids[i]] in exclude_class: continue
            if scores[i] < min_thres: continue
            
            y1, x1, y2, x2 = boxes[i]
            x, y, w, h = x1, y1, x2-x1, y2-y1
            area_rect=w*h/(im_width*im_height)
            confidence = scores[i]
            area = areas_mask[i]
            d_class[label][0] = d_class[label][0]+1
            d_class['graffiti_mask'][0] = d_class['graffiti_mask'][0] + area
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(label, x, y, w, h,float(confidence), area_rect*100, area*100))
          f.close()
        
        img_annotations = pd.DataFrame(data=d_class)
        data_objects = data_objects.append(img_annotations, ignore_index=True)
      print('***********************')
      pbar.update(1)
      
      if not dont_show:
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                  classes, r['scores'], ax=ax, exclude=exclude_class,
                                  scorethresh=min_thres)

        fig.savefig(outdir+file_name, bbox_inches='tight', pad_inches = 0)

    data_objects.to_csv(outdir+'objects_detected.csv', index=False)
############################################################
#  Training
############################################################

def getGraffitiDetector(weights_path=ROOT_DIR+"mask_rcnn_graffiti_0029.h5", output_dir=''):
  print('Loading model...')
  config = InferenceConfig(nclasses=NUM_CLASSES)
  config.display()
  
  model = modellib.MaskRCNN(mode="inference", config=config, model_dir=output_dir)
  model.load_weights(weights_path, by_name=True)
  return model

def getGraffitiDetections(images_dir, model, output_dir=''):
  verifyDir(output_dir)
  print('***************************')
  print('**** GRAFFITI GETECTION ****')
  print('***************************')
  batch_img(model, images_dir, outdir=output_dir)
  return

if __name__ == '__main__':
    #model = getGraffitiDetector()
    #os.exit()

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect graffitis.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'batch'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/graffiti/dataset/",
                        help='Directory of the Graffiti dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--imdir', help='Images path')
    parser.add_argument('--outdir', help='CSV output directory')
    parser.add_argument('--reverse', action='store_true', help='Reverse order to traverse files')
    parser.add_argument('--shuffle', action='store_true', help='Shuffled order')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video or args.imdir,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GraffitiConfig(nclasses=NUM_CLASSES)
    else:
        config = InferenceConfig(nclasses=NUM_CLASSES)
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    #if args.weights.lower() == "coco":
    if 'coco' in args.weights.lower():
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print('If you want to change the number of classes, consider removing the last layers.')
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.logs)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, imdir=args.imdir,
                                video_path=args.video)
    elif args.command == "batchimg":
        batch_img(model, args.imdir, args.outdir, args.reverse, args.shuffle)
    elif args.command == "batchcsv":
        batchcsv(model, args.imdir, args.outdir, args.reverse, args.shuffle)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
