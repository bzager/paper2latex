# segment.py
# Ben Zager
# Image segmentation module for paper2latex
# Converts image of handwriting into a binary image
# requires tools.py

# 1) Preprocess image with Gaussian smoothing
# 2) Convert to binary image with sauvola threshold
# 3) Clean binary image with morphological operations
# 4) Calculate properties of each segmented region


import tools

# Preprocess image
# Gaussian smoothing with given sigma
def preprocess(img,sig=2.0):
	img = tools.smooth(img,sig=sig)
	return img

# convert grayscale to binary using sauvola threshold
def threshold(img,size=25,k=0.2):
	thresh = tools.sauvola(img,size=size,k=k)
	return img > thresh

# Post process image for to clean up binarization
# Apply erosion/opening with structuring element of size [radius]
# Remove small holes with fewer than [size] pixels
def clean(img,radius=3,method="",size=32):
	if method == "erode":
		img = tools.erode(img,radius)
	elif method == "open":
		img = tools.opening(img,radius)
	elif method == "dilate":
		img = tools.dilate(img,radius)
	elif method == "close":
		img = tools.closing(img,radius)

	return tools.removeHoles(img,size=size)

# Convert grayscale [img] to binary image
def binarize(img,radius,method,sig):
	img = preprocess(img,sig=sig)
	img = threshold(img)
	img = clean(img,radius=radius,method=method)
	return img


# Label each region and caluculate properties
def segmentProperties(img,buff=1):
	labels = tools.label(img)
	labels = tools.clearBorder(labels,buff=buff)
	return labels,tools.properties(labels)

# Run all steps
def segmentation(img,radius=3,method="open",sig=1.0,buff=1):
	img = binarize(img,radius,method,sig)
	labels,props = segmentProperties(img,buff=buff)

	return img,labels,props


