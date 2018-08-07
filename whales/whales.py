import pandas as pd
import os
import tensorflow as tf

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

train_data_path = "./train/"
train_file_path = "/home/pbreton/.kaggle/competitions/whale-categorization-playground/train.csv"
test_data_path = "./test/"
image_size = (64, 64)
batch_size = 16

train = pd.read_csv(train_file_path).values

categories = set(train[:, 1])

def move_everything_to_classes(train):
    for image, category in train:
        if not os.path.exists(os.path.join(train_data_path, category)): os.mkdir(
            os.path.join(train_data_path, category))
        if not os.path.exists(os.path.join(train_data_path, category, image)):
            os.rename(os.path.join(train_data_path, image), os.path.join(train_data_path, category, image))

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, input_shape=(*image_size, 3), activation='relu', kernel_size=(5, 5), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='relu'))
model.add(Dense(len(categories)-1, activation='softmax'))

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10)