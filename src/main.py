# main.py
# Ben Zager
# Main file for paper2latex

import sys
import argparse


from tools import loadScaled,displayAll
from segment import segmentation,binarize
from symbol import Symbol
from extract import prepPhogs,prepImgs

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray' # good colormaps: prism,flag

# 
def main():
	args = getArgs()
	symbols = runSegment()

	
# get command line arguments
# image name and maximum size for images segmentation
def getArgs():
	fname = sys.argv[1]

	if len(sys.argv) == 3: 
		maxsize = sys.argv[2]
	else: 
		maxsize = None
	return fname,maxsize


# run segmenation on an image
def runSegment():
	fname,maxsize = getArgs()

	print("loading image...")
	img = loadScaled(fname,maxsize=maxsize)

	radius = 4
	method = "open"
	sig = 2.0

	print("segmenting...")
	bimg,labels,props = segmentation(img,radius=radius,method=method,sig=sig)
	symbols = [Symbol(region,img) for region in props[1:]]

	print("displaying...")
	displayAll([sym.hogImg for sym in symbols])
	displayAll([sym.lbp for sym in symbols])
	plt.tight_layout(pad=0.1,h_pad=None,w_pad=None)

	return symbols


if __name__=="__main__":
	main()
	plt.show()

