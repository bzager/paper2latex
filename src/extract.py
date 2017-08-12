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


# 
def main():
	root = "../train/"
	dirname = "images/"

	subdirs = getArgs()

	if subdirs[0] == "ALL":
		extractPhogAll(root,direc)

	else:
		for subdir in subdirs:
			extractDir(root,dirname,subdir)


# parse command line arguments
def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("fnames",nargs="+")
	args = parser.parse_args()
	
	return args.fnames

# calculates and saves Phog features for all training images 
# of a single class
def extract(subdir,fname):
	phogname = os.path.splitext(fname)[0] # remove .jpg
	
	if os.path.isfile("../train/phog/"+subdir+"/"+phogname+".npy"):
		print("    "+phogname+" (already saved)")
		return np.load("../train/phog/"+subdir+"/"+phogname+".npy")

	img = load(fname,"../train/images/"+subdir)
	sym = Symbol(props=None,img=np.invert(img))
	sym.calcPhog()
	
	print("    "+phogname)
	sym.savePhog(subdir,phogname)

	return sym.phog

# calculates and saves Phog features of training image
# returns matrix of dimension (num,phogs.size)
def extractDir(root,dirname,subdir,num=None):
	print(subdir)
	if not os.path.isdir(root+dirname+subdir) or subdir == ".DS_Store":
		return

	phogs = []
	for fname in os.listdir(root+dirname+subdir)[:num]:
		phog = extract(subdir,fname)
		phogs.append(phog)

	return np.asarray(phogs)

"""
# calculates and saves Phog features of all training images 
def extractAll(root,direc):
	for dirname in os.listdir(root+direc):
		phogs = extractDir(root,direc,dirname)
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
# dimensions are (num*len(subdirs),phog.size)
# labels is vector of integer label for each row
def prepPhogs(subdirs,num,form="int",root="../train/",dirname="images/"):	
	allPhogs = []
	labels = []
	labelDict = getLabels(subdirs)

	for subdir in subdirs:
		phogs = extractDir(root,dirname,subdir,num)
		allPhogs.append(phogs)
		labels += [labelDict[subdir] for i in range(phogs.shape[0])]

	return np.concatenate(allPhogs),prepLabels(labels,form)

# creates array of images
# (num*len(subdirs),45,45)
def prepImgs(subdirs,num,form="int",root="../train/",dirname="images/"):
	allImgs = []
	labels = []
	labelDict = getLabels(subdirs)

	for subdir in subdirs:
		imgs = loadAll(directory=root+dirname+subdir,count=num)
		allImgs.append(np.asarray(imgs))
		labels += [labelDict[subdir] for i in range(allImgs[-1].shape[0])]

	return np.concatenate(allImgs),prepLabels(labels,form)



if __name__=="__main__":
	main()
