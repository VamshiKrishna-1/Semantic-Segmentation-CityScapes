import tensorflow as tf
from utils import get_paths, compare_overlapped_images
from model import UNet
import random


if __name__ == "__main__":

    # Getting all the paths required from get_paths function in utils module
    img_paths, label_paths, val_img_paths, val_label_paths = get_paths()

    # Initializing Model
    model = UNet()

    # Loding weights into the model
    model.load_weights('224x448.h5')

    # Selecting a random sample image from validation data
    num = random.randint(0, len(val_img_paths) - 1)
    compare_overlapped_images(model, val_img_paths[num], val_label_paths[num]).compare()
