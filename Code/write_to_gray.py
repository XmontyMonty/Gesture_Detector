import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from random import randint
import cv2

file_path = 'D:\\Users\\Master\\Documents\\A.I Project\\TensorFlow' \
            '\\TensorFlow-GPU\\Git_Repo\\Gesture_Detector\\Image_Data\\train\\'
write_path = 'D:\\Users\\Master\\Documents\\A.I Project\\TensorFlow' \
             '\\TensorFlow-GPU\\Git_Repo\\Gesture_Detector\\Image_Data\\train\\'
categories = ['Closed_Fist\\', 'Open_Hand\\', 'Other\\', 'Swipe_Left\\',
              'Swipe_Right\\']

for cat in categories:
    counter = 0
    image_name = cat[:-1] + '{}.png'
    read = file_path + cat
    write = write_path + cat
    image = cv2.imread(read + image_name.format(counter))
    while type(image) != type(None):
        print(image_name.format(counter))
        image = cv2.resize(image, (128, 128))
        cv2.imwrite(write + image_name.format(counter), image)
        counter += 1
        image = cv2.imread(read + image_name.format(counter))
print('Finished')
#
# counter = 0
# while counter <= 50:
#     read = file_path + categories[0]
#     write = write_path + categories[0]
#     image_name = categories[0][:-1] + '{}.png'
#     image = cv2.imread(read + image_name.format(counter))
#     print(cv2.imwrite(write + image_name.format(counter), image))
#     counter += 1
# print('Finished')
