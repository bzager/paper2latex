# tools.py
# Ben Zager
# Image processing functions for paper2latex
# used by: binzarize.py, segment.py

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform,filters,morphology,exposure,measure


#########################################################
########################## I/O ##########################
#########################################################

# loads grayscale image from given directory
def load(name,directory=""):
	return io.imread("../images/"+name,as_grey=True)

# loads image, scaling it to below max number of pixels
def loadScaled(name,directory="",maxsize=1000000):
	img = load(name,directory)
		
	if img.size > maxsize:
		scale = np.sqrt(maxsize/float(img.size))
		return rescale(img,scale) 
	
	return img

# loads all images from a directory
def loadAll(directory="",maxsize=1000000):
	imgs = []
	
	for fname in os.listdir("../images/"):
		if fname==".DS_Store":
			continue
		imgs.append(loadScaled(fname,directory,maxsize=maxsize))
	
	return imgs

#########################################################
##################### Preprocessing #####################
#########################################################

# rescales image by a given size
def rescale(img,scale):
	scaled = transform.rescale(img,scale,mode='reflect',preserve_range=True)
	return scaled

# adjusts image intensities to enhance contrast
def equalize(img):
	#img = exposure.adjust_sigmoid(img,cutoff=0.5,gain=5,inv=False)
	img = exposure.adjust_gamma(img,gamma=1.0,gain=1.0)
	
	return img

#########################################################
###################### Thresholding #####################
#########################################################

# 
def sauvola(img,size=25,k=0.2):
	return filters.threshold_sauvola(img,window_size=size,k=k)

#########################################################
##################### Morphology ####################
#########################################################

def getSelem(radius):
	return morphology.disk(radius)

def opening(img,radius):
	return morphology.binary_opening(img,selem=getSelem(radius))

def erode(img,radius):
	return morphology.binary_erosion(img,selem=getSelem(radius))

def removeHoles(img,size=32):
	return morphology.remove_small_holes(img,min_size=size)


#########################################################
###################### Segmentation #####################
#########################################################

def label(img):
	return measure.label(img,connectivity=2,background=1)

def properties(labels):
	return measure.regionprops(labels)


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
def displayAll(imgs,title=""):
	cols = int(np.ceil(np.sqrt(len(imgs))))
	rows = int(np.ceil(len(imgs) / cols))

	fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(12,6))
	plt.figtext(0.5,0.9,title)

	for im,ax in zip(imgs,axes.flatten()):
		ax.imshow(im)
		ax.set_xticks([]); ax.set_yticks([])

#
def getShape(bins):
	center = (bins[:-1] + bins[1:]) / 2
	width = 1.0*(bins[1] - bins[0])
	return center,width

#
def hist(data,nbins=256,range=None):
	his,bins = np.histogram(data,bins=nbins,density=True)
	center,width = getShape(bins)
	return his,center,width

#
def histPlot(hist,center,width,title=""):
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,2))
	ax.bar(center,hist,width=width)
	ax.set_title(title)
	#ax.set_xlim([0,1])
	#ax.set_ylim([0,0.1])

# plots an image and histogram
def plotImgHist(img,hist,center,width,text=" "):
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
	ax[0].imshow(img); ax[0].set_xticks([]); ax[0].set_yticks([]);
	ax[1].bar(center,hist,width=width)
	#ax[1].set_xlim([-1,1])
	#ax[1].set_ylim([0,0.1])
	
	ax[1].set_title(text)

def fullImgHist(img,data,text=""):
	h,cen,wid = hist(data)
	plotImgHist(img,h,cen,wid,text=text)

#
def fullHist(data,title="",nbins=256,range=None):
	h,cen,wid = hist(data,nbins=nbins,range=range)
	histPlot(h,cen,wid,title=title)
	return h
