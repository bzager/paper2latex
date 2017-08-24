# extract.py
# Ben Zager
# Extacts and saves PHOG features from training images

# Run:
# python extract.py [names of symbols to extract]
# Can use any amount of names

# Loads all images of those symbols and calculates the phog features
# Saved as numpy binary files (.npy) in ../train/phog/[symbol name]

# Use ALL as name to extract all 

import sys
import os
import argparse
import time

import numpy as np
from tools import load,loadAll
from symbol import Symbol


root = "../train/"
imgDir = "images/"
phogDir = "phog/"
ALL = "ALL" # input name to extract phogs for all symbols

letters = "abcdefghijklmnopqrstuvwxyz"
lowercase = list(letters)
uppercase = list(letters.upper())
integers = [str(i) for i in range(10)]

# 
def main():
	names = getArgs()

	if names[0] == ALL:
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

"""
# calculates and saves Phog features of all training images 
def extractAll():
	for name in os.listdir(root+imgDir):
		phogs = extractDir(name)
"""

# returns a list of all symbols names in training directory
def getAllNames():
	return os.listdir(root+imgDir)

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

# does in place random reordering of phog data
def shufflePhog(phogs,labels):
	toShuffle = np.append(data,temp_labels,axis=1)
	np.random.shuffle(toShuffle)

	return toShuffle[:,:-1],toShuffle[:,-1].astype(np.int)

# 
def shuffleImgs(imgs,labels):
	inds = np.random.permutation(labels.size)
	return imgs[inds,:,:],labels[inds]


# returns a random permutation of num elements of a list
def permute(lst,num):
	inds = np.random.permutation(len(lst))[:num]
	return lst[inds]

# gets indices for separating test data
def getInds(num,numClass,numTest):
	inds = []

	for i in range(numClass):
		for j in range(numTest):
			inds.append(i*num + j)

	return inds

# calculates and saves Phog features for all training images 
# of a single class
def extract(name,fname):
	phogname = os.path.splitext(fname)[0] # remove .jpg
	
	if os.path.isfile(root+phogDir+name+"/"+phogname+".npy"):
		return np.load(root+phogDir+name+"/"+phogname+".npy")

	img = load(fname,root+imgDir+name)
	sym = Symbol(props=None,img=np.invert(img))
	sym.calcPhog()
	
	print("    "+phogname+" saved")
	sym.savePhog(name,phogname)

	return sym.phog

# calculates and saves Phog features of training image
# returns matrix of dimension (num,phogs.size)
# can either be first num samples, or num random samples
def extractDir(name,num=None,random=False):
	path = root+imgDir+name
	if not os.path.isdir(path) or name == ".DS_Store":
		return

	phogs = []
	fnames = os.listdir(path)
	if num != None and num >= len(fnames): 
		num = None

	#if random: files = permute(fnames,num)
	#else: files = fnames[:num]

	files = fnames[:num]

	print(name+" "+str(len(fnames)))
	for fname in files:
		if fname == ".DS_Store": continue
		phog = extract(name,fname)
		phogs.append(phog)

	return np.asarray(phogs)


# creates stack of phog vectors for all training data
# dimensions are (num*len(names),phog.size)
# labels is vector of integer label for each row
def prepPhogs(names,num,random=True):	
	allPhogs = []
	labels = []
	labelDict = getLabels(names)

	for name in names:
		phogs = extractDir(name,num,random=random)
		if phogs is None:
			continue
		allPhogs.append(phogs)
		labels += [labelDict[name] for i in range(phogs.shape[0])]

	return np.concatenate(allPhogs),np.asarray(labels)

# creates array of images
# (num*len(names),45,45)
def prepImgs(names,num):
	allImgs = []
	labels = []
	labelDict = getLabels(names)

	for name in names:
		imgs = loadAll(directory=root+imgDir+name,count=num)
		allImgs.append(np.asarray(imgs))
		labels += [labelDict[name] for i in range(allImgs[-1].shape[0])]
		print(name+" "+str(len(imgs))+" "+str(num))
		print(allImgs[-1].shape)

	return np.concatenate(allImgs),np.asarray(labels)


# divides the loaded data into training and test sets
# also shuffles the order of the training data
# numTest is the number of test samples from each class
# labels must be in integer format
def getTest(data,labels,num,numTest):
	numClass = np.unique(labels).size # number of unique classes
	inds = getInds(num,numClass,numTest) # list of indices to make tests

	testLabels = labels[inds]

	if data.ndim == 2: # phog data
		testData = data[inds,:]
	elif data.ndim == 3:
		testData = data[inds,:,:]

	trainData = np.delete(data,inds,axis=0)
	trainLabels = np.delete(labels,inds,axis=0)
	
	return trainData,trainLabels,testData,testLabels


# initializes phog data for classification
# loads data, splits into train and test data, shuffles train data 
def initPhogs(names,numTrain,numTest,random=True):
	num = numTrain+numTest
	phogs,labels = prepPhogs(names,num,random=random)
	trainPhogs,trainLabels,testPhogs,testLabels = getTest(phogs,labels,num,numTest)
	
	#trainPhogs,trainLabels = shufflePhogs(trainPhogs,trainLabels)

	return trainPhogs,trainLabels,testPhogs,testLabels

# initializes image data for classification
def initImgs(names,numTrain,numTest):
	num = numTrain+numTest
	imgs,labels = prepImgs(names,num)
	trainImgs,trainLabels,testImgs,testLabels = getTest(imgs,labels,num,numTest)
	
	#trainImgs,trainLabels = shuffleImgs(trainImgs,trainLabels)

	return trainImgs,trainLabels,testImgs,testLabels


if __name__=="__main__":
	main()
