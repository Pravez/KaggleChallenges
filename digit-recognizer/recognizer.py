import pandas as pd
from keras import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
train_file_path = "/home/pbreton/.kaggle/competitions/digit-recognizer/train.csv"
test_file_path = "/home/pbreton/.kaggle/competitions/digit-recognizer/test.csv"

load_from_file = False


train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

Y_train = train["label"].values
X_train = train.drop(labels = ["label"], axis=1).values
X_test = test.values

X_train = X_train/255.
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test / 255.
X_test = X_test.reshape(-1, 28, 28, 1)

if load_from_file:
    model = load_model('recognizer.h5')
else:
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    model = Sequential()
    model.add(Conv2D(64, input_shape=(28, 28, 1), kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, activation='relu', kernel_size=(3, 3), padding="same"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, activation='relu', kernel_size=(3, 3), padding="same"))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, activation='relu', kernel_size=(3, 3), padding="same"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

    Y_train = to_categorical(Y_train, num_classes=10)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train)/32, epochs=5)
    model.save('recognizer.h5')

results = model.predict(X_test, verbose=1)
result = np.argmax(results, axis=1)

np.savetxt('submission3.csv', np.c_[range(1, len(test)+1), result], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
#sub = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
#sub.to_csv("mnist_datagen.csv", index=False)