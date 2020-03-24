import numpy as np
import mahotas
from skimage.feature import greycomatrix, greycoprops

ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
OFFSETS = [1]

PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

image = np.array([[0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 2, 2, 2],
                   [2, 2, 3, 3]], dtype=np.uint8)
                   
glcm = greycomatrix(image, OFFSETS, ANGLES, levels=4)#, normed=True)

print(glcm[:, :, 0,0],"\n")

for p in PROPERTIES:
  prop = greycoprops(glcm, p)
  print("* "+p+": ", prop)

for i in range(len(ANGLES)):
  haralick = mahotas.features.haralick(glcm[:,:,0,i])
  print(haralick)
