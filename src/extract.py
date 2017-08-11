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
	

	if os.path.isfile("../train/phog/"+fname):
		print("    "+phogname+" (already saved)")
		return np.load("../train/phog/"+fname)

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

# creates dictionary of phogs matrices
# keys are symbol names, values are phog matrices
def preparePhogs(subdirs,num,root="../train/",dirname="images/"):	
	allPhogs = {}

	for subdir in subdirs:
		phogs = extractDir(root,dirname,subdir,num)
		allPhogs[subdir] = phogs

	return allPhogs

# creates dictionary of images
# each values is an array of dimension (num,45,45)
def prepareImgs(subdirs,num,root="../train/",dirname="images/"):
	allImgs = {}

	for subdir in subdirs:
		imgs = loadAll(directory=root+dirname+subdir,count=num)
		allImgs[subdir] = np.asarray(imgs)

	return allImgs



if __name__=="__main__":
	main()
