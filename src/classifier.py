# classifier.py
# Ben Zager
# Test classifier on PHOG features or original images

# Run:
# python classifier.py [# training imgs for each class] [# test imgs for each class]

import sys
import cProfile

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import extract
from extract import initPhogs,initImgs


def main():
	numTrain = int(sys.argv[1])
	numTest = int(sys.argv[2])
	num = numTrain + numTest
	names = extract.lowercase #+ extract.uppercase # symbols to use

	trainPhogs,trainLabels,testPhogs,testLabels = initPhogs(names,numTest,numTrain)
	trainImgs,trainLabels,testImgs,testLabels = initImgs(names,numTest,numTrain)

	C = 0.5
	gamma = "auto"

	results = runSVM(trainPhogs,trainLabels,testPhogs,C=C,gamma=gamma)
	accuracy = getAccuracy(results,testLabels)
	print(np.around(accuracy,2))


# calculate accuracy 
# compares output labels with known labels
def getAccuracy(results,labels):
	correct = np.sum(np.equal(labels,results))
	return float(correct) / labels.size

# 
def runSVM(phogs,labels,testPhogs,C=1.0,gamma="auto"):
	clf = OneVsRestClassifier(SVC(C=C,gamma=gamma))
	clf.fit(phogs,labels)
	results = clf.predict(testPhogs)

	return results


if __name__=="__main__":
	main()
	#cProfile.run("main()")



