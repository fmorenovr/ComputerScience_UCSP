from keras.preprocessing import image as image_utils
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import logging
import cv2
from scipy.io import savemat
from scipy.io import loadmat
#from mat4py import loadmat
#import h5py
#import hdf5storage
#import tables

#mat = tables.open_file('tmp_decaf_filelist.mat')
#mat = hdf5storage.loadmat('tmp_decaf_filelist.mat')
mat = loadmat('features/tmp_decaf_filelist.mat')

print ('loading model...')
logging.getLogger().setLevel(logging.INFO)
net = VGG16(weights="imagenet")
net.summary()

extractfeatures = Model(input=net.input, output=net.get_layer('fc2').output)

images = mat['filelist'][0];
print("Images has lenght of: " + str(len(images)))

scores = []
features = []
for i in range(0, len(images)):
  print ('classifying image %d'%(i+1))
  img = image_utils.load_img(images[i][0], target_size=(224, 224))
  img = image_utils.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  preds = net.predict(img)
  #P = decode_predictions(preds)
  scores.append(preds[0])

  fc_features = extractfeatures.predict(img)
  fc_features = fc_features.reshape(1,4096)
  features.append(fc_features[0])

savemat('features/tmp_decaf_output.mat', mdict={'scores': scores, 'features': features})
