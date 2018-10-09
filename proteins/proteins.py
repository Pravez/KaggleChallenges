import os

import keras.backend as K
import pandas as pd
import numpy as np

from data_gen import DataGen
from model import model

from keras import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 4
SHAPE = (299, 299, 3)
STEPS_PER_EPOCH = 100
EPOCHS = 2

train_path = "./res/train"

train_data = pd.read_csv("./res/train.csv")
train_dataset_info = []

for name, labels in zip(train_data['Id'], train_data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(train_path, name),
        'labels': np.array([int(label) for label in labels])
    })
train_dataset_info = np.array(train_dataset_info)

# images, labels = next(train_datagen)
# fig, ax = plt.subplots(1,5,figsize=(25,5))
# for i in range(5):
#    ax[i].imshow(images[i])
# print('min: {0}, max: {1}'.format(images.min(), images.max()))
# fig.show()

model = model((299, 299, 3), output_shape=28)

model.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3), metrics=["acc"])

checkpointer = ModelCheckpoint('./res/model/inceptionv4.model', verbose=2, save_best_only=True)
tensorboard = TensorBoard(log_dir='./res/logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
                          write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                          embeddings_metadata=None, embeddings_data=None)
np.random.seed(1996)

indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)

train_indexes = indexes[:int(len(indexes) * 0.8)]
valid_indexes = indexes[int(len(indexes) * 0.8):]

train_datagen = DataGen.create_set(train_dataset_info[train_indexes], BATCH_SIZE, SHAPE, augment=True)
valid_datagen = DataGen.create_set(train_dataset_info[valid_indexes], BATCH_SIZE, SHAPE, augment=False)

history = model.fit_generator(
    train_datagen,
    validation_data=next(valid_datagen),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpointer, tensorboard]
)


submit = pd.read_csv('./res/sample_submission.csv')

predicted = []
for name in tqdm(submit["Id"]):
    path = os.path.join('./res/test/', name)
    image = DataGen.load_image(path, SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('./res/submission.csv', index=False)