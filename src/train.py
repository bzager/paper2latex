# train.py
# Charles Winston
# Neural network module for training character classifier

from keras.models import Sequential
from keras.layers import Dense, Activation

def train_sequential(input_dim, dropout_rate):
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
                  metrics=['accuracy']))

    # get x and y

    # train model
    model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

    # test model
    score = model.evaluate(x_test, y_test, batch_size=128)
