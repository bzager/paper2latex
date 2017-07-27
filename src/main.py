# main.py
# Ben Zager
# Main file for paper2latex

import sys
sys.dont_write_bytecode = True

from tools import loadScaled,displayAll
from segment import binarize,segmentProperties
from symbol import Symbol

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# 
def main():
	print("loading image...")
	fname,maxsize = getInput()
	img = loadScaled(fname,maxsize=maxsize)

	radius = 4
	method = "open"
	#method "erode"

	print("binarizing image...")
	bimg = binarize(img,radius,method)

	print("calculating properties...")
	props,labels = segmentProperties(bimg)
	symbols = [Symbol(region) for region in props[1:]]
	
	print("displaying...")
	displayAll([bimg,labels])
	displayAll([sym.image for sym in symbols])


# 
def getInput():
	fname = sys.argv[1]

	if len(sys.argv) == 3: 
		maxsize = sys.argv[2]
	else: 
		maxsize = 1000000
	return fname,maxsize


if __name__=="__main__":
	main()
	plt.show()

