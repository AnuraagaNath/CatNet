from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import joblib
from glob import glob
from preprocessing import getTrainValidData

# load model
model = keras.models.load_model('./CatNet/')

# load class indices
class_indices = joblib.load('./class_indices.joblib')

# divide the data in different class in the folder
test1, test2 = getTrainValidData('./images')


print('Test1 image count: ', test1.n)
print('Test2 image count: ', test2.n)
test1_loss, test1_accuracy = model.evaluate(test1)
test2_loss, test2_accuracy = model.evaluate(test2)

print('='*100)
print(f'Test1 Loss: {test1_loss:.4f}')
print(f'Test1 Accuracy: {test1_accuracy:.4f}\n')
print(f'Test2 Loss: {test2_loss:.4f}')
print(f'Test2 Accuracy: {test2_accuracy:.4f}')
print('='*100)