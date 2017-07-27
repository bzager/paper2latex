# segment.py
# Ben Zager
# Image segmentation module for paper2latex
# Converts image of handwriting into a binary image
# requires tools.py

# 1) Convert to binary image with sauvola threshold
# 2) Clean binary image with morphological operations
# 3) Calculate properties of each segmented region


import tools

# convert grayscale to binary using sauvola threshold
def threshold(img,size=25,k=0.2):
	thresh = tools.sauvola(img,size=size,k=k)
	return img > thresh

# Post process image for to clean up binarization
def clean(img,radius=1,method=""):
	if method == "erode":
		img = tools.erode(img,radius=radius)
	elif method == "open":
		img = tools.opening(img,radius=radius)

	return tools.removeHoles(img)

# run each step
def binarize(img,radius,method):
	img = threshold(img)
	img = clean(img,radius,method)
	return img

#
def segmentProperties(img):
	labels = tools.label(img)
	return tools.properties(labels),labels
