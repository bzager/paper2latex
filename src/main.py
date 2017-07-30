# main.py
# Ben Zager
# Main file for paper2latex

import sys
sys.dont_write_bytecode = True

from tools import loadScaled,displayAll
from segment import segmentation,binarize
from symbol import Symbol

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray' # good colormaps: prism,flag

# 
def main():
	print("loading image...")
	fname,maxsize = getInput()
	img = loadScaled(fname,maxsize=maxsize)

	radius = 4
	method = "open"
	sig = 2.0

	print("segmenting...")
	bimg,labels,props = segmentation(img,radius=radius,method=method,sig=sig)
	symbols = [Symbol(region,img) for region in props[1:]]

	print("displaying...")
	#displayAll([img,bimg])
	#displayAll([sym.image for sym in symbols])
	#displayAll([sym.original for sym in symbols])
	displayAll([sym.hogImg for sym in symbols])
	displayAll([sym.lbp for sym in symbols])

# get image name and maximum size
def getInput():
	fname = sys.argv[1]

	if len(sys.argv) == 3: 
		maxsize = sys.argv[2]
	else: 
		maxsize = None
	return fname,maxsize


if __name__=="__main__":
	main()
	plt.tight_layout(pad=0.1,h_pad=None,w_pad=None)
	plt.show()

