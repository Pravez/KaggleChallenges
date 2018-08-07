from keras import Sequential, Model, Input
from keras.applications import densenet
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np

train_json_path = "/home/pbreton/.kaggle/competitions/imaterialist-challenge-furniture-2018/train.json"
test_json_path = "/home/pbreton/.kaggle/competitions/imaterialist-challenge-furniture-2018/test.json"
path_train = './resources/train/'
path_test = './resources/test/'
path_validation = './resources/validation/'

classes = set([d['label_id'] for d in json.load(open(train_json_path))['annotations']])
test_images_list = [item['image_id'] for item in json.load(open(test_json_path))['images']]

image_rows, image_cols, image_channels = 221, 221, 3
batch_size = 8

steps_per_epoch = 100
epochs = 50
validation_steps = 50

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=10,
        data_format='channels_last',
        horizontal_flip=True,
        vertical_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(path_train, target_size=(image_rows, image_cols), batch_size=batch_size, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(path_validation, target_size=(image_rows, image_cols), batch_size=batch_size, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(path_test, target_size=(image_rows, image_cols), batch_size=batch_size)

    return train_generator, validation_generator, test_generator

def create_model():
    model = Sequential()

    model.add(Conv2D(32, input_shape=(image_rows, image_cols, image_channels), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['acc'])

    return model

if __name__ == '__main__':

    dense = densenet.DenseNet121(include_top=False, weights=None, input_shape=(image_rows, image_cols, image_channels), pooling='max', classes=len(classes))
    x = Flatten()(dense.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=dense.inputs, outputs = output)
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['acc'])

    train_gen, valid_gen, test_gen = create_data_generators()

    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=valid_gen, validation_steps=validation_steps, shuffle=True)
    model.save('model.h5')

    results = model.predict_generator(test_gen)


    np.savetxt('submission.csv', np.c_[range(1, len(test_images_list) + 1), results.astype('int')], delimiter=',', header='id,predicted',
               comments='', fmt='%d')