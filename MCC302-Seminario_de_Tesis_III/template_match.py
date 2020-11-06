#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

visualize = False

template = cv2.imread('test/template.png',0)
template = cv2.Canny(template, 50, 200)
tW, tH = template.shape[::-1]

img_rgb = cv2.imread('test/img_1.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#threshold = 0.8
#loc = np.where( res >= threshold)
#for pt in zip(*loc[::-1]):
#  cv2.rectangle(img_rgb, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
#cv2.imwrite('res.png',img_rgb)

# loop over the images to find the template in
found = None
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
	# resize the image according to the scale, and keep track
	# of the ratio of the resizing
	resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale))
	r = img_gray.shape[1] / float(resized.shape[1])
	# if the resized image is smaller than the template, then break
	# from the loop
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break

	# detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = cv2.Canny(resized, 50, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	# check to see if the iteration should be visualized
	if not visualize:
		# draw a bounding box around the detected region
		clone = np.dstack([edged, edged, edged])
		cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
			(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		cv2.imshow("Visualize", clone)
		cv2.waitKey(0)
	# if we have found a new maximum correlation value, then update
	# the bookkeeping variable
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)
# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# draw a bounding box around the detected result and display the image
cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", img_rgb)
cv2.waitKey(0)
