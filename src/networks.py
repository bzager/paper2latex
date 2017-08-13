# train.py
# Charles Winston
# Neural network module for training character classifier

import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras import metrics

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
    
    input_dim = len(phogs[0])
    num_classes = len(names)
    batch_size = 32
    dropout_rate = 0
    epochs = 1

    model = simple_model(input_dim,dropout_rate, num_classes)
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
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# convolutional network
#def convolutional_model(input_dim, dropout_rate):
    # TODO


# train model
def train_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, int2OneHot(y_train), epochs=epochs, batch_size=batch_size)

    return model

def test_model(model, x_test, y_test, batch_size):
    # test model
    return model.evaluate(x_test, int2OneHot(y_test), batch_size=batch_size)


if __name__=="__main__":
    main()
