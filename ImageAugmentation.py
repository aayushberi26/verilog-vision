import os
import sys

from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img
)

class ImageAugmentation:
    def __init__(self, source_dir, destination_dir, num_augmentations=5):
        self.datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                zca_whitening=True,
                fill_mode='nearest'
            )
        self.image_paths = [source_dir + "/" + path for path in os.listdir(source_dir)]
        self.destination_dir = destination_dir
        self.num_augmentations = num_augmentations

    def augment_using_keras(self):
        '''
        Adapted from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        '''
        for image_path in self.image_paths:
            if '.png' in image_path or '.jpg' in image_path:
                img = load_img(image_path)  # PIL image
                x = img_to_array(img)  # Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 3, 150, 150)
                # generates batches of randomly transformed images
                i = 0
                for batch in self.datagen.flow(x, batch_size=1,
                                          save_to_dir=self.destination_dir, save_prefix='gate', save_format='jpeg'):
                    i += 1
                    if i > self.num_augmentations:
                        break  # otherwise the generator would loop indefinitely

if __name__ == '__main__':
    ia = ImageAugmentation(sys.argv[1], sys.argv[2])
    ia.augment_using_keras()