# train.py
# Charles Winston
# Neural network module for training character classifier

import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from extract import prepPhogs
from classifier import getTest

def main():
    numTrain = int(sys.argv[1])
    numTest = int(sys.argv[2])
    num = numTrain + numTest
    subdirs = [str(i) for i in range(0,10)]

    # get training and testing data
    phogs, labels = prepPhogs(subdirs, num, form="oh")
    trainPhogs, trainLabels, testPhogs, testLabels, inds = getTest(phogs,labels,num,numTest)

    input_dim = len(phogs[1])
    model = simple_model(input_dim, 0)
    model = train_model(trainPhogs, trainLabels, 1, 32)
    score = test_model(testPhogs, testLabels, 32)

    print('Accuracy: ')
    print(score)

# a simple neural network
def simple_model(input_dim, dropout_rate):
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
    model.add(Dense(10,
                    activation='softmax',
                    use_bias='true',
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros',
                    ))
    model.add(Dropout(dropout_rate))


    # configure learning process
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    return model

# convolutional network
#def convolutional_model(input_dim, dropout_rate):
    # TODO


# train model
def train_model(x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size)

    return model

def test_model(x_test, y_test, batch_size):
    # test model
    return model.evaluate(x_test, y_test, batch_size=batch_size)


if __name__=="__main__":
    main()
