import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import class_dict

# Defining DataGenerator Class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths , mask_paths , batch_size , img_size=(224, 448), shuffle=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_img_paths = [self.img_paths[k] for k in indexes]
        batch_mask_paths = [self.mask_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_img_paths, batch_mask_paths)

        return X, y

    def on_epoch_end(self):
        'Shuffles indexes after each epoch if shuffle is set to True'
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def one_hot_encode(self, mask, rgb_to_class):
        # Determine number of unique classes
        unique_classes = len(set(rgb_to_class.values()))

        # Initialize an empty array with shape (height, width, num_unique_classes)
        one_hot = np.zeros(mask.shape[:2] + (unique_classes,))

        for rgb, idx in rgb_to_class.items():
            matching_pixels = (mask == rgb).all(axis=-1)  # Find pixels in the mask that match this RGB value
            one_hot[..., idx] += matching_pixels  # Increment by 1 where matches are found

        # Ensure that no pixel is counted more than once
        one_hot = np.clip(one_hot, 0, 1)

        return one_hot


    def __data_generation(self, batch_img_paths, batch_mask_paths):
        'Generates data containing batch_size samples'
    

        X = np.empty((len(batch_img_paths) * 2, 224, 448, 3))

        num_classes = len(set(class_dict.values()))
        y = np.empty((len(batch_mask_paths) * 2, 224, 448, num_classes))

        for i, (img_path, mask_path) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            img = cv2.imread(img_path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            # Resize the image to 512x1024
            img = tf.image.resize(img, (224, 448)).numpy()
            img = img.astype(np.float16) / 255.0

            mask = cv2.imread(mask_path, 1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = tf.image.resize(mask, (224, 448), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
    

            # Original image and mask
            X[i * 2] = img
            y[i * 2] = self.one_hot_encode(mask, class_dict)

            # Flip the image and mask horizontally
            flipped_img = np.fliplr(img)
            flipped_mask = np.fliplr(mask)

            # Add the flipped image and mask to the batch
            X[(i * 2) + 1] = flipped_img
            y[(i * 2) + 1] = self.one_hot_encode(flipped_mask, class_dict)

        return X, y