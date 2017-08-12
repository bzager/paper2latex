# extract.py
# Ben Zager
# Extacts and saves PHOG features from training images

# Run:
# python extract.py [names of symbols to extract]
# Can use any amount of names

# Loads all images of those symbols and calculates the phog features
# Saved as numpy binary files (.npy) in ../train/phog/[symbol name]

# Use * as name to extract all 

import sys
import os
import argparse

import numpy as np
from tools import load,loadAll
from symbol import Symbol


root = "../train/"
imgDir = "images/"
phogDir = "phog/"

# 
def main():
	names = getArgs()

	if names[0] == "ALL":
		extractAll()

	else:
		for name in names:
			extractDir(name)


# parse command line arguments
def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("names",nargs="+")
	args = parser.parse_args()
	
	return args.names

# calculates and saves Phog features for all training images 
# of a single class
def extract(name,fname):
	phogname = os.path.splitext(fname)[0] # remove .jpg
	
	if os.path.isfile(root+phogDir+name+"/"+phogname+".npy"):
		return np.load(root+phogDir+name+"/"+phogname+".npy")

	img = load(fname,root+imgDir+name)
	sym = Symbol(props=None,img=np.invert(img))
	sym.calcPhog()
	
	print("    "+phogname+"saved")
	sym.savePhog(name,phogname)

	return sym.phog

# calculates and saves Phog features of training image
# returns matrix of dimension (num,phogs.size)
def extractDir(name,num=None):
	print(name)
	if not os.path.isdir(root+imgDir+name) or name == ".DS_Store":
		return

	phogs = []
	for fname in os.listdir(root+imgDir+name)[:num]:
		phog = extract(name,fname)
		phogs.append(phog)

	return np.asarray(phogs)

"""
# calculates and saves Phog features of all training images 
def extractAll():
	for name in os.listdir(root+imgDir):
		phogs = extractDir(name)
"""

# creates dict mapping category names to integer labels
def getLabels(names):
	return dict(zip(names,np.arange(len(names))))


# convert integer label vector to stack of one-hot vectors
# output: matrix where each row is a one-hot vector
def int2OneHot(vec):
	labels = np.zeros([vec.size,np.amax(vec)+1])
	labels[np.arange(vec.size),vec] = 1
	return labels

# converts 
def oneHot2Int(oneHot):
	return np.where(oneHot)[0][0]

# 
def prepLabels(labels,form):
	labels = np.asarray(labels)
	
	if form == "oh":
		return int2OneHot(labels)

	return labels


# creates stack of phog vectors for all training data
# dimensions are (num*len(names),phog.size)
# labels is vector of integer label for each row
def prepPhogs(names,num,form="int"):	
	allPhogs = []
	labels = []
	labelDict = getLabels(names)

	for name in names:
		phogs = extractDir(root,imgDir,name,num)
		allPhogs.append(phogs)
		labels += [labelDict[name] for i in range(phogs.shape[0])]

	return np.concatenate(allPhogs),prepLabels(labels,form)

# creates array of images
# (num*len(names),45,45)
def prepImgs(names,num,form="int"):
	allImgs = []
	labels = []
	labelDict = getLabels(names)

	for name in names:
		imgs = loadAll(directory=root+imgDir+name,count=num)
		allImgs.append(np.asarray(imgs))
		labels += [labelDict[name] for i in range(allImgs[-1].shape[0])]

	return np.concatenate(allImgs),prepLabels(labels,form)



if __name__=="__main__":
	main()
