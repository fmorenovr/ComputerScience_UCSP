import random
import cv2
import numpy as np

def rotate(img):
  sign=random.randint(-20,20)%2
  if sign==0:
    sign=-1   

  col = img.shape[0]    
  row = img.shape[1]    
  angle=(random.randint(1,100)%60)*sign
  img_center=tuple(np.array([row,col])/2)
  rot_mat = cv2.getRotationMatrix2D(img_center, angle,1)
  result = cv2.warpAffine(img, rot_mat, (col,row), flags=cv2.INTER_LINEAR)
  return result

def erosion(img,kernel):
  return cv2.erode(img,kernel,iterations = 1)

def dilation(img,kernel):
  return cv2.dilate(img,kernel,iterations = 2)
 
def opening(img,kernel):
  return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
 
def closing(img,kernel):
  return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
 
def gradient(img,kernel):
  return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
 
def morphology(img):
  dade   = random.randint(1,2)    
  kernel = np.ones((2,2),np.uint8)
  switcher = {
      1: erosion(img,kernel),
      2: dilation(img,kernel)
  #    3: opening(img,kernel),
  #    4: closing(img,kernel),
  #    5: gradient(img,kernel),
  }
  result = switcher.get(dade, lambda: "Invalid")
  return result

def scale(img):
  height=28
  width=28
  height = ( ( random.randint(1,2000) % 20 ) * -1 ) + height;
  width  = ( ( random.randint(1,2000) % 20 ) * -1 ) + width;
  result = cv2.resize(img,(height,width),interpolation=cv2.INTER_CUBIC)
  return result 

def traslation(img):
  rows,cols = img.shape
  pts1 = np.float32([[0,0],[0,rows],[cols,0],[rows,cols]])
  varWidth=rows/2
  varHeight=cols/2
  widthWarp=rows  + random.randint(-int(varWidth/1.0), int(varWidth/1.0))
  heightWarp=cols + random.randint(-int(varHeight/1.0),int(varHeight/1.0))
  
  pts2 = np.float32([[0,0],[0,rows],[cols,0],[widthWarp,heightWarp]])

  M = cv2.getPerspectiveTransform(pts1,pts2)

  dst = cv2.warpPerspective(img,M,(28,28))
  return dst


def addLines(image, chr_num):
  numLines=random.randint(1, 4) 
  
  W = image.shape[0]
  for x in range(0,numLines):
    R     = 255#random.randint(0,255)
    G     = 255#random.randint(0,255)
    B     = 255#random.randint(0,255)
    
    startX = random.randint(0, W*chr_num) % image.shape[1];
    endX   = random.randint(0, W*chr_num) % image.shape[1];
    startY = random.randint(0, W*chr_num) % image.shape[0];
    endY   = random.randint(0, W*chr_num) % image.shape[0];
    lines  = cv2.line(image,(startX,startY),(endX,endY), (R,G,B) , random.randint(10, 15))
  return lines 

def sp_noise(image,prob):
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

def addNoise(image,chr_num):   
  numNoise = random.randint(120, 150) 
  W = image.shape[0]
  for x in range(0,numNoise):
    R     = 255#random.randint(0,255)
    G     = 255#random.randint(0,255)
    B     = 255#random.randint(0,255)

    i = random.randint(0, W)  % image.shape[0];
    j = random.randint(0, W*chr_num)% image.shape[1];
    radius = random.randint(10,15)
    noise  = cv2.circle(image,(j,i),radius, (R,G,B), -1)      
  return noise
