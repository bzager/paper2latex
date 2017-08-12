# classifier.py
# Ben Zager
# SVM classifier on PHOG features

import sys

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from tools import displayAll
from extract import prepPhogs,prepImgs


# divides the loaded data into training and test sets
# numTest is the number of test samples from each class
def getTest(phogs,labels,num,numTest):
	numClass = np.amax(labels) # number of unique classes
	inds = [] # list of indices to make tests

	for i in range(numClass):
		for j in range(numTest):
			inds.append(i*num + j)

	testPhogs = phogs[inds,:]
	testLabels = labels[inds]

	phogs = np.delete(phogs,inds,axis=0)
	labels = np.delete(labels,inds,axis=0)
	
	return phogs,labels,testPhogs,testLabels,inds

# 
def getAccuracy(labels,results):
	correct = np.sum(np.equal(labels,results))
	return float(correct) / labels.size



if __name__=="__main__":

	numTrain = int(sys.argv[1])
	numTest = int(sys.argv[2])
	num = numTrain + numTest
	names = [str(i) for i in range(0,10)]

	phogs,labels = prepPhogs(names,num)
	imgs,labels = prepImgs(names,num)
	phogs,labels,testPhogs,testLabels,inds = getTest(phogs,labels,num,numTest)

	C = 1.0
	gamma = "auto"

	clf = OneVsRestClassifier(SVC(C=C,gamma=gamma))
	clf.fit(phogs,labels)

	results = clf.predict(testPhogs)
	accuracy = getAccuracy(testLabels,results)
	
	print(testLabels)
	print(results)
	print(np.around(accuracy,2))



