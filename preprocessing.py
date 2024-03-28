from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from glob import glob
import PIL
import joblib


# import tensorflow as tf
# tf.config.list_physical_devices()


filepath = './PetImages'


# Data Augmentation
image_gen = ImageDataGenerator(
        rescale = 1/255.0,
        rotation_range = 20,
        zoom_range = 0.05,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        shear_range = 0.05,
        horizontal_flip = True,
        fill_mode = "nearest",
        validation_split = 0.20)



# function to get train and validation data using the data augmentation. This method retrieves the data straight from the directory. 
def getTrainValidData(filepath, target_size=(100,100), batch_size=64, class_mode='binary'):

    # training data generation
    train_generator = image_gen.flow_from_directory(
        directory=filepath,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training',
        shuffle=True,
        seed=42
    )

    # validation data generation
    valid_generator = image_gen.flow_from_directory(
    directory=filepath,
    target_size=target_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=class_mode,
    subset='validation',
    shuffle=True,
    seed=42
    )
    return train_generator, valid_generator




# # removing corrupted files

# PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# def process_images(image_files):
#     for image_file in image_files:
#         try:
#             with PIL.Image.open(image_file):
#                 pass
#         except PIL.UnidentifiedImageError:
#             print(f"Can't identify image file {image_file}. Deleting.")
#             os.remove(image_file)

# process_images(glob(dog_path + '/*'))
# process_images(glob(cat_path + '/*'))

