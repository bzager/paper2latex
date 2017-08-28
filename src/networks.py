# train.py
# Charles Winston
# Neural network module for training character classifier

import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers.convolutional import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D
from keras import regularizers

import extract
from extract import initPhogs,initImgs,int2OneHot,oneHot2Int

# 
def main():
	numTrain = int(sys.argv[1])
	numTest = int(sys.argv[2])
	num = numTrain + numTest
	names = extract.lowercase

	num_classes = len(names)
	batch_size = 32
	dropout_rate = 0.1
	epochs = 4
	size = "small" # size of CNN

	# get training and testing data
	trainPhogs,trainLabels,testPhogs,testLabels = initPhogs(names,numTest,numTrain)
	#trainImgs,trainLabels,testImgs,testLabels = initImgs(names,numTest,numTrain)
	
	phog_dim = trainPhogs[0].shape
	#img_dim = trainImgs[0].shape
	hidden_size = 32

	model = simple_model(phog_dim,hidden_size,dropout_rate,num_classes)
	#model = conv_img(img_dim,dropout_rate,num_classes,size=size)
	#model = conv_phog(phog_dim,dropout_rate,num_classes)

	model = train_model(model,trainPhogs,trainLabels,epochs,batch_size)
	score = test_model(model,testPhogs,testLabels,batch_size)

	print("\n")
	print("Loss: "+str(np.around(score[0],2)))
	print("Accuracy: "+str(np.around(score[1],2)))
	print("\n")
	

# a simple neural network
def simple_model(input_shape,hidden_size,dropout_rate,num_classes):
	kreg = 0.01
	breg = 0.01

	model = Sequential()
	model.add(Dense(hidden_size,input_shape=input_shape,activation='relu',
								kernel_initializer='random_uniform',
								bias_initializer='zeros',
								kernel_regularizer=regularizers.l2(kreg),
								bias_regularizer=regularizers.l2(breg)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(num_classes,activation='softmax')) # output
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	return model

# convolutional network using 2d binary image input
# machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# small architecture: 
# input->Conv(32,5x5,ReLU)->Pool->Dropout->Dense(128,ReLU)->Dense(numclass,softmax)
# big architecture: 
# input->Conv(32,5x5,ReLU)->Pool->Conv(16,3x3,ReLU)->Pool->Dropout->Dense(128,ReLU)->Dense(50,ReLU)->Dense(numclass,softmax)
def conv_img(input_shape,dropout_rate,num_classes,size='small'):
	
	num_filters = 32
	ker_size = 5

	model = Sequential()
	model.add(Conv2D(num_filters,(ker_size,ker_size),input_shape=input_shape,padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	if size == 'small': 
		model = add_small(model)
	elif size == 'big': 
		model = add_big(model)

	model.add(Dense(num_classes,activation='softmax')) # output
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return model

# 
def add_small(model):
	model.add(Dropout(dropout_rate))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))

	return model

# 
def add_big(model):
	model.add(Conv2D(16,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model = setup_small(model)
	model.add(Dense(50,activation='relu'))

	return model

# convolutional network using 1d phog vector input
def conv_phog(input_shape,dropout_rate,num_classes):
	model = Sequential()
	#model.add(Conv1D())

	# TODO

	return model


# train model
def train_model(model,x_train,y_train,x_test,y_test,epochs,batch_size):
	model.fit(x_train,int2OneHot(y_train),validation_data=(x_test,int2OneHot(y_test)),epochs=epochs,batch_size=batch_size,shuffle=True)

	return model

# test model
def test_model(model,x_test,y_test,batch_size):
	return model.evaluate(x_test,int2OneHot(y_test),batch_size=batch_size)

# saves model weights as HDF5 file
def save_weights(model,path,name):
	model.save_weights(path+name)

# loads saved model weights
def load_weights(model,path,name):
	return model.load_weights(path+name)


# prints and returns info about the model
def get_info(model):
	summ = model.summary()
	weights = model.get_weights()

	return summ,weights

# approximates size of hidden layer
# scale should be in range 2-10
# stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
def calcHiddenSize(trainData,num_classes,scale=5.0):
	num_samples = trainData.shape[0]
	input_size = trainData.shape[1]
	return int(np.floor(num_samples / (scale*(input_size+num_classes))))


if __name__=="__main__":
	main()
