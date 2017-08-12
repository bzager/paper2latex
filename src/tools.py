# tools.py
# Ben Zager
# Image processing functions for paper2latex
# used by: segment.py, main.py
#

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,util,transform,filters,morphology,feature,measure,segmentation


#########################################################
########################## I/O ##########################
#########################################################

# loads grayscale image from given directory
def load(name,directory=""):
	return io.imread(directory+"/"+name,as_grey=True)

# loads image, scaling it to below max number of pixels
def loadScaled(name,directory="",maxsize=1000000):
	img = load(name,directory)

	if maxsize == None:
		return img

	if img.size > maxsize:
		scale = np.sqrt(maxsize/float(img.size))
		return rescale(img,scale)

	return img

# loads all images from a directory
def loadAll(directory="",maxsize=400000,count=20):
	imgs = []

	for fname in os.listdir(directory+"/")[:count]:
		if fname==".DS_Store":
			continue
		imgs.append(loadScaled(fname,directory+"/",maxsize=maxsize))

	return imgs

# intensity limits for img,
# should be (0,1) or (0,255)
def limits(img):
	return util.dtype_limits(img)

#########################################################
##################### Preprocessing #####################
#########################################################

# rescales image by a given factor
def rescale(img,scale):
	return transform.rescale(img,scale,mode='reflect',preserve_range=True)

# pads img with zeros
# width ->  ((before axis 0,after axis 0),(before axis 1, after axis 1)),
def pad(img,width=((0,0),(0,0))):
	return util.pad(img,width,mode="constant",constant_values=0)

# pads an image along smaller axis to make square
# if both axes are even or odd, pad (0,0) to larger
# if one axis is even and other is odd, pad (1,0) to
def squareImg(img):
	diff = np.amax(img.shape)-np.amin(img.shape)
	padMain = (int(np.ceil(diff/2)),int(np.ceil(diff/2)))

	if img.shape[0] == img.shape[1]:
		return img

	if img.shape[0] % 2 != img.shape[1] % 2:
		padFix = (0,1)
	else:
		padFix = (0,0)

	if img.shape[0] > img.shape[1]:
		return pad(img,((padFix,padMain)))
	elif img.shape[1] > img.shape[0]:
		return pad(img,(padMain,padFix))
	else:
		return img

#
def padExtra(img,extra):
	width = (int(np.ceil(extra/2)),int(np.floor(extra/2)))
	return pad(img,width=(width,width))

# resizes and pads image to given size
def resizeImg(img,size=45,extra=5):
	square = squareImg(img)
	scale = size / np.sqrt(square.size)
	rescaled = rescale(square,scale) # size x size
	return padExtra(rescaled,extra) # (size+extra) x (size+extra)


# Gaussian smoothing
def smooth(img,sig=1):
	if sig == None:
		return img
	return filters.gaussian(img,sigma=sig)


#########################################################
###################### Thresholding #####################
#########################################################

# sauvola thresholding
def sauvola(img,size=25,k=0.2):
	return filters.threshold_sauvola(img,window_size=size,k=k)

#########################################################
##################### Morphology ####################
#########################################################

# structuring element
# disk of given radius
def getSelem(radius):
	return morphology.disk(radius)

# morphological opening for binary image
def opening(img,radius):
	return morphology.binary_opening(img,selem=getSelem(radius))

#
def closing(img,radius):
	return morphology.binary_closing(img,selem=getSelem(radius))

# morphological erosion for binary image
def erode(img,radius):
	return morphology.binary_erosion(img,selem=getSelem(radius))

# morphological dilation for binary image
def dilate(img,radius):
	return morphology.binary_dilation(img,selem=getSelem(radius))

# remove small holes smaller than [size] pixels
def removeHoles(img,size=32):
	return morphology.remove_small_holes(img,min_size=size)

#########################################################
###################### Segmentation #####################
#########################################################

"""
# Felzenszwalb's efficient graph based image segmentation (EGBIS)
def felz(img,scale=1,sig=0.8,min_size=20):
	return segmentation.felzenszwalb(img,scale=scale,sigma=sig,min_size=min_size)
"""

# removes regions with [buff] pixels of border
def clearBorder(labels,buff=1):
	return segmentation.clear_border(labels,buffer_size=buff)

# labels each isolated region with an integer
def label(img):
	return measure.label(img,connectivity=1,background=1)

# calculate properties of each labeled region
def properties(labels):
	return measure.regionprops(labels)

#########################################################
################### Feature Extraction ##################
#########################################################


# histogram of oriented gradients
# returns tuple (feature vector, visualization w/ same size as input)
def calcHOG(img,orientations=8,cell=(5,5),block=(1,1),vector=True):
	return feature.hog(img,orientations=orientations,pixels_per_cell=cell,cells_per_block=block,block_norm="L2-Hys",visualise=True,feature_vector=vector)

# local binary pattern
# R = radius of neighbor pixels, P = number of pixels at that radius
# methods: default, ror, uniform, nri_uniform, var
def lbp(img,P=8,R=1,method="uniform"):
	return feature.local_binary_pattern(img,P=P,R=R,method=method)


#########################################################
######################## Display ########################
#########################################################

# display 2 images side by side
def display(img1,img2,titles=[]):
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
	im1 = ax[0].imshow(img1)
	im2 = ax[1].imshow(img2)
	ax[0].set_xticks([]); ax[0].set_yticks([]); ax[1].set_xticks([]); ax[1].set_yticks([]);

	if len(titles) != 0:
		ax[0].set_title(titles[0])
		ax[1].set_title(titles[1])
	if len(img1.shape) == 2:
		fig.colorbar(im1,ax=ax[0],fraction=0.046,pad=0.04)
	if len(img2.shape) == 2:
		fig.colorbar(im2,ax=ax[1],fraction=0.046,pad=0.04)


# display a list of images
def displayAll(imgs,title="",cmap="gray"):
	cols = int(np.ceil(np.sqrt(len(imgs))))
	rows = int(np.ceil(len(imgs) / float(cols)))

	fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(20,12))
	plt.figtext(0.5,0.9,title)

	while len(imgs) < len(axes.flatten()):
		imgs.append(np.ones([10,10]))

	for im,ax in zip(imgs,axes.flatten()):
		ax.imshow(im,cmap=cmap)
		ax.set_xticks([]); ax.set_yticks([])
	#for ax in axes.flatten():
		#ax.set_xticks([]); ax.set_yticks([])


# gets histogram shape from [bins]
def getShape(bins):
	center = (bins[:-1] + bins[1:]) / 2
	width = 1.0*(bins[1] - bins[0])
	return center,width

# calculates histogram
def hist(data,nbins=256,range=None):
	his,bins = np.histogram(data,bins=nbins,density=True)
	center,width = getShape(bins)
	return his,center,width

# plots a given histogram
def histPlot(hist,center,width,title=""):
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,2))
	ax.bar(center,hist,width=width)
	ax.set_title(title)
	#ax.set_xlim([0,1])
	#ax.set_ylim([0,0.1])

# plots an image and histogram
def plotImgHist(imgs,hist,center,width,text=" "):
	fig,axes = plt.subplots(nrows=1,ncols=len(imgs)+1,figsize=(16,8))

	for ax,im in zip(axes,imgs):
		ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]);

	axes[-1].bar(center,hist,width=width)
	axes[-1].set_xlim([0,hist.size])
	axes[-1].set_ylim([0,np.amax(hist)*1.05])
	axes[-1].set_title(text)
	axes[-1].set_aspect(axes[-1].get_xlim()[1]/axes[-1].get_ylim()[1])


# calculates histogram of data and displays next to an image
def fullImgHist(img,data,text=""):
	h,cen,wid = hist(data)
	plotImgHist(img,h,cen,wid,text=text)

# calculates and plots histogram of [data]
def fullHist(data,title="",nbins=256,range=None):
	h,cen,wid = hist(data,nbins=nbins,range=range)
	histPlot(h,cen,wid,title=title)
	return h
