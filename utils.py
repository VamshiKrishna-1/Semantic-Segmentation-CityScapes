import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np


# Function to return paths of images and labels of train and test data

def get_paths():
    # Define the root directory for images and labels
    root_dir = "../Data/Semantic-Segmentation-of-CityScapes"
    img_dir = f"{root_dir}/leftImg8bit/train/*/*png"
    label_dir = f"{root_dir}/gtFine/train/*/*_color.png"

    val_img_dir = f"{root_dir}/leftImg8bit/val/*/*.png"
    val_label_dir = f"{root_dir}/gtFine/val/*/*_color.png"


    # Get all image and label paths
    img_paths = tf.io.gfile.glob(img_dir)
    label_paths = tf.io.gfile.glob(label_dir)

    val_img_paths = tf.io.gfile.glob(val_img_dir)
    val_label_paths = tf.io.gfile.glob(val_label_dir)

    # Sorting the images to maintain the correct order
    img_paths.sort()
    label_paths.sort()

    val_img_paths.sort()
    val_label_paths.sort()

    return (img_paths, label_paths, val_img_paths, val_label_paths)


# Since I am only interested in classifying certain labels, I am going to consider only certain labels are create a python dictionary
class_dict = {  (  0,  0,  0): 0,
                (  0,  0,  0): 0,
                (  0,  0,  0): 0,
                (  0,  0,  0): 0,
                (  0,  0,  0): 0,
                (111, 74,  0): 0,
                ( 81,  0, 81): 0,
                (128, 64,128): 0,
                (244, 35,232): 0,
                (250,170,160): 0,
                (230,150,140): 0,
                ( 70, 70, 70): 1,
                (102,102,156): 1,
                (190,153,153): 1,
                (180,165,180): 1,
                (150,100,100): 1,
                (150,120, 90): 1,
                (153,153,153): 2,
                (153,153,153): 2,
                (250,170, 30): 2,
                (220,220,  0): 2,
                (107,142, 35): 2,
                (152,251,152): 2,
                ( 70,130,180): 0,
                (220, 20, 60): 3,
                (255,  0,  0): 3,
                (  0,  0,142): 4,
                (  0,  0, 70): 4,
                (  0, 60,100): 4,
                (  0,  0, 90): 4,
                (  0,  0,110): 4,
                (  0, 80,100): 4,
                (  0,  0,230): 4,
                (119, 11, 32): 4,
                (  0,  0,142): 4
}


class compare_overlapped_images():
    def __init__(self, model, input_image_path, label_image_path):
        """It is required the input size to be:
        1. Input image size >= (448, 896)
        2. Input height to width ratio = 1:2"""
        self.input_image_path = input_image_path
        self.label_image_path = label_image_path
        self.model = model

    def compare(self):
        # Get preprocessed input image
        input_image = self._load_preprocess_image()
        
        # Predict images
        predicted_image = self._predict_image(input_image)

        # Get Actual mask
        rgb_mask_image = self._get_label_image()

        # One Hot Encode the mask_image
        ohe_mask_image = compare_overlapped_images._one_hot_encode_image(rgb_mask_image, class_dict)

        # Again converting to RGB that is defined by us(OUR RGB Values)
        rgb_mask = self._one_hot_to_rgb(ohe_mask_image, True)
        

        # Generating Overlayed Image predicted
        overlayed_image_pred = compare_overlapped_images._overlayed_image(first_image = input_image, second_image = predicted_image)

        # Generating Overlayed Image actual
        overlayed_image_actual = compare_overlapped_images._overlayed_image(first_image = input_image, second_image = rgb_mask)        

        # Plotting the image
        compare_overlapped_images._plot_image(overlayed_image_pred, 'Overlayed Image Predicted')

        # Plotting the image
        compare_overlapped_images._plot_image(overlayed_image_actual, "Overlayed Image Actual")
    


    def _overlayed_image(first_image, second_image, alpha = 0.6):
        # Convert to numpy arrays
        first_image = first_image.numpy()
        second_image = second_image.numpy()

        # Converting to float32 values
        second_image = second_image.astype('float32')/255.0

        # Applying overlay on both images
        overlayed_image = cv2.addWeighted(first_image, 1, second_image, alpha, 0)

        return overlayed_image
        
    def _load_preprocess_image(self):
        # Reading image
        img = tf.io.read_file(self.input_image_path)
        img = tf.image.decode_png(img, channels=3)

        # Normalize
        img = tf.cast(img, tf.float32)

        img = img/225.0

        # Resize
        img = tf.image.resize(img, (224, 448))
        
        return img

    
    def _plot_image(image, title):
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    def _get_label_image(self):
        img = tf.io.read_file(self.label_image_path)
        img = tf.image.decode_png(img, channels=3)
        
        # Normalize
        img = tf.cast(img, tf.float32)
        
        # Assuming `img` is your mask tensor
        img = tf.image.resize(img, [224, 448], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        img = img.numpy()
        
        return img

    def _one_hot_encode_image(image, class_dict):
        # Assuming class_dict maps RGB to a single class index and there are 5 classes in total.
        num_classes = 5
        height, width, _ = image.shape
        one_hot_encoded = np.zeros((height, width, num_classes), dtype=np.float32)
    
        # Iterate over each unique RGB value in the class dictionary.
        for rgb, class_index in class_dict.items():
            # Create a mask for pixels that match the current RGB value.
            mask = (image == rgb).all(axis=-1)
    
            # Set the corresponding location in the one-hot encoded array to 1.
            one_hot_encoded[mask, class_index] = 1
    
        return one_hot_encoded

    
    def _one_hot_to_rgb(self, one_hot_tensor, mask = False):
        """
        Converts an one-hot encoded image tensor to an RGB image tensor.

        Parameters:
        - one_hot_tensor: The one-hot encoded image tensor of shape [height, width, channels].
        - channel_to_rgb_dict: A dictionary mapping channel indices to RGB values.

        Returns:
        - An RGB image tensor of shape [height, width, 3].
        """

        channel_to_rgb_dict = { 0: (225,  225,  225),
                        1: (250,170,0),
                        2: (0,153,153) ,
                        3: (220,9, 225),
                        4: (222,22,222)
        }

        # Reshape the tensor to (256, 256, 3)

        if mask:
            pass
        else:   
            one_hot_tensor = tf.squeeze(one_hot_tensor, axis=0)
            
        # Get the channel index with the highest value for each pixel (argmax across the depth)
        channel_indices = tf.argmax(one_hot_tensor, axis=-1)

        # Convert the dictionary to a tensor of RGB values
        rgb_tensor = tf.constant([channel_to_rgb_dict[i] for i in range(5)])

        # Use tf.gather to map channel indices to RGB values
        rgb_image = tf.gather(rgb_tensor, channel_indices)
        
        return rgb_image



    def _predict_image(self, image):
        image = tf.expand_dims(image, axis=0)  # Adds a batch dimension
        predicted_layers = self.model.predict(image)
        decoded_image = self._one_hot_to_rgb(predicted_layers)

        return decoded_image



