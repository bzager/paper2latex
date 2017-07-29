# segment.py
# Ben Zager
# Image segmentation module for paper2latex
# Converts image of handwriting into a binary image
# requires tools.py

# 1) Convert to binary image with sauvola threshold
# 2) Clean binary image with morphological operations
# 3) Calculate properties of each segmented region


import tools

def preprocess(img,sig=2.0):
	img = tools.smooth(img,sig=sig)
	return img

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

	img = tools.removeHoles(img)

	return img

# 
def binarize(img,radius,method,sig):
	img = preprocess(img,sig=sig)
	img = threshold(img)
	img = clean(img,radius=radius,method=method)
	return img


# label each region and caluculate properties
def segmentProperties(img,buff=1):
	labels = tools.label(img)
	labels = tools.clearBorder(labels,buff=buff)
	return labels,tools.properties(labels)

# run each step
def segmentation(img,radius=3,method="open",sig=1.0,buff=1):
	img = binarize(img,radius,method,sig)
	labels,props = segmentProperties(img,buff=buff)

	return img,labels,props


