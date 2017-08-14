# train.py
# Charles Winston
# Neural network module for training character classifier

import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers.convolutional import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D

import extract
from extract import initPhogs,initImgs,int2OneHot,oneHot2Int

# 
def main():
	numTrain = int(sys.argv[1])
	numTest = int(sys.argv[2])
	num = numTrain + numTest
	names = extract.integers

	# get training and testing data
	trainPhogs,trainLabels,testPhogs,testLabels = initPhogs(names,numTest,numTrain,random=True)
	#trainImgs,trainLabels,testImgs,testLabels = initImgs(names,numTest,numTrain)
	
	phog_dim = phogs.shape[1]
	#img_dim = imgs.shape[1:]
	num_classes = len(names)
	batch_size = 32
	dropout_rate = 0
	epochs = 1
	size = "small"

	model = simple_model(phog_dim, dropout_rate, num_classes)
	#model = convolutional_Img(img_dim, dropout_rate, num_classes, size=size)
	#model = convolutional_Phog(phog_dim, dropout_rate, num_classes)

	model = train_model(model,trainPhogs,trainLabels,epochs,batch_size)
	score = test_model(model,testPhogs,testLabels,batch_size)

	print("\n")
	print("Loss: "+str(np.around(score[0],2)))
	print("Accuracy: "+str(np.around(score[1],2)))
	

# a simple neural network
def simple_model(input_dim, dropout_rate, num_classes):
	# build the network
	model = Sequential()
	model.add(Dense(32,
					input_dim=input_dim,
					activation='relu',
					use_bias='true',
					kernel_initializer='random_uniform',
					bias_initializer='zeros',
					))
	model.add(Dropout(dropout_rate))
	model.add(Dense(num_classes,
					activation='softmax',
					use_bias='true',
					kernel_initializer='random_uniform',
					bias_initializer='zeros',
					))
	model.add(Dropout(dropout_rate))

	# configure learning process
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	return model

# convolutional network using 1d phog vector input
def convolutional_Phog(input_dim, dropout_rate, num_classes):
	model = Sequential()
	#model.add(Conv1D())

	# TODO

	return model

# convolutional network using 2d binary image input
def convolutional_Img(input_dim, dropout_rate, num_classes, size="small"):
	# Build
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_dim=input_dim, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	if size == "small":
		model.add(Dropout(dropout_rate))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))

	elif size == "large":
		model.add(Conv2D(16, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(dropout_rate))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(50, activation='relu'))
	
	# Output layer
	model.add(Dense(num_classes, activation='softmax')) 

	# Compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


# train model
def train_model(model, x_train, y_train, epochs, batch_size):
	model.fit(x_train, int2OneHot(y_train), epochs=epochs, batch_size=batch_size, shuffle=True)

	return model

def test_model(model, x_test, y_test, batch_size):
	# test model
	return model.evaluate(x_test, int2OneHot(y_test), batch_size=batch_size)


if __name__=="__main__":
	main()
