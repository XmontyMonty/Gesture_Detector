import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

camera = cv2.VideoCapture(0)

while True:
    na, image = camera.read()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv hue sat value
    lower_skin = np.array([20, 20, 30])
    upper_skin = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('frame', image)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
cv2.destroyAllWindows()
camera.release()
