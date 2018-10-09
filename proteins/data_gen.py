import numpy as np
import skimage.io
import skimage.transform

from imgaug import augmenters as iaa


LABELS = 28

class DataGen:
    @staticmethod
    def create_set(dataset_info, batch_size, shape, augment=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, LABELS))
            for i, index in enumerate(random_indexes):
                image = DataGen.load_image(dataset_info[index]['path'], shape)

                if augment:
                    image = DataGen.augment(image)

                batch_images[i] = image
                batch_labels[i][dataset_info[index]['labels']] = 1

            yield batch_images, batch_labels


    @staticmethod
    def load_image(path, shape):
        image_red = skimage.io.imread(path+'_red.png')
        image_yellow = skimage.io.imread(path+'_yellow.png')
        image_blue = skimage.io.imread(path+'_blue.png')
        image_green = skimage.io.imread(path+'_green.png')

        image_red += (image_yellow/2).astype(np.uint8)
        image_green += (image_yellow/2).astype(np.uint8)

        image = np.stack((image_red, image_green, image_blue), -1)
        image = skimage.transform.resize(image, (shape[0], shape[1]), mode="reflect")
        return image

    @staticmethod
    def augment(image):
        augmented_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])], random_order=True)

        image_aug = augmented_img.augment_image(image)
        return image_aug

