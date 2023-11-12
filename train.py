import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from tensorflow.keras.callbacks import TensorBoard
import datetime

from dataset import DataGenerator
from utils import class_dict, get_paths, compare_overlapped_images
from model import UNet
from loss import DiceLoss, DiceCELoss

# Getting all the paths required from get_paths function in utils module
img_paths, label_paths, val_img_paths, val_label_paths = get_paths()

# Defining Batch_size
batch_size = 6

# Step 1: Define a generator function
def train_data_gen():
    data_generator = DataGenerator(img_paths = img_paths , mask_paths = label_paths , batch_size = batch_size)
    for data in data_generator:
        yield data
        

def val_data_gen():
    data_generator = DataGenerator(img_paths = val_img_paths , mask_paths = val_label_paths , batch_size = batch_size)
    for data in data_generator:
        yield data

# Step 2: Convert the generator to tf.data.Dataset
num_unique_classes = len(set(class_dict.values()))

train_dataset = tf.data.Dataset.from_generator(
    train_data_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 448, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 448, num_unique_classes), dtype=tf.uint8)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    val_data_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 448, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 448, num_unique_classes), dtype=tf.uint8)
    )
)

def plot_samples(number_of_samples = 1):
    """Make sure that the number of samples is less than (batch_size * 2)"""
    
    assert number_of_samples <= (batch_size * 2), "Please provide number of samples less than (batch_size * 2)"
    
    # 1. Extract one batch
    for inputs, outputs in train_dataset.take(1):
        pass

    for sample_num in range(number_of_samples):
        sample_input = inputs[sample_num].numpy()
        sample_output = outputs[sample_num].numpy()
        
        
        # 2. Plot the input image
        plt.figure(figsize=(6, 6))
        plt.imshow((sample_input))
        plt.title("Input Image")
        plt.axis("off")
        plt.show()
        
        # 3. Plot each channel of the output as a grayscale image
        num_channels = sample_output.shape[-1]  # assuming the shape is (height, width, channels)
        fig, axes = plt.subplots(1, num_channels, figsize=(20, 20))
        
        for i in range(num_channels):
            channel_image = sample_output[:, :, i]
            ax = axes[i]
            ax.imshow(channel_image, cmap='gray')
            ax.set_title(f"Output Channel {i+1}")
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    plot_samples(2)

    # Create a U-Net model instance
    unet_model = UNet()

    # Compile the model
    unet_model.compile(optimizer='adam', loss =DiceLoss())

    PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE  # Let TensorFlow determine the optimal buffer size
    train_dataset = train_dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

    unet_model.build(input_shape = ((None, 224, 448, 3)))

    # Display the model summary
    unet_model.summary()

    # Training the model
    history = unet_model.fit(
        train_dataset,
        epochs=5,
        validation_data = val_dataset
    )
    # Save the model
    unet_model.save('model.h5')  # saves the model architecture, weights, and training config

    # Convert the history.history dict to a JSON file
    with open('history.json', 'w') as f:
        json.dump(history.history, f)

