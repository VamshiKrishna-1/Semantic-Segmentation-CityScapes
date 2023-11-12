# Semantic-Segmentation-CityScapes
This project is dedicated to the semantic segmentation of cityscapes dataset using a UNet model implemented in TensorFlow. It includes various modules for dataset handling, loss function computation, model architecture, and training, offering a complete framework for sophisticated image segmentation tasks.

# Introduction
This suite of Python scripts is specifically designed for the task of semantic segmentation on the cityscapes dataset using the UNet model, a powerful architecture in the realm of image processing. The project leverages TensorFlow to efficiently handle and process complex image data, aiming to deliver precise segmentation results.

# File Descriptions
- <b>dataset.py</b>: Handles operations related to the cityscapes dataset, including loading, preprocessing, and batch generation for segmentation tasks.
- <b>loss.py</b>: Implements the loss computation functions crucial for training the segmentation model.
- <b>model.py</b>: Defines the UNet architecture tailored for semantic segmentation of urban scenes.
- <b>train.py</b>: Facilitates the training of the UNet model on the cityscapes dataset, including routines for managing training and validation data.
- <b>utils.py</b>: Offers a suite of utility functions, supporting image processing, data visualization, and performance evaluation tasks.


# Usage
To use this project for semantic segmentation tasks:
 - Ensure you have TensorFlow installed and configured.
 - Prepare the cityscapes dataset in the required format.
 - Run the train.py script to commence training the model.
 - Use the model for segmentation predictions on urban scene images.

# Aftermath of training
![Training and Validation Loss Graph]((https://imgur.com/a/F51TnqN))

# Contributor
Vamshi Krishna Tinberiveni

# License
This project is licensed under the MIT License.
