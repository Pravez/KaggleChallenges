from keras import Sequential
from keras.layers import Flatten, Activation, Dense, Dropout

from inception_v4 import create_model


def model(input_shape, output_shape):

    inception = create_model(include_top=False, weights='imagenet', input_shape=input_shape)

    inception.trainable = False

    model = Sequential()
    model.add(inception)
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.33))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.33))
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    return model