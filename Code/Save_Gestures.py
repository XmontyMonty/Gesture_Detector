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
import cv2


cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
temp_image = None
cat = "Other"
path = 'D:\\Users\\Master\\Documents\\A.I Project\\TensorFlow' \
       '\\TensorFlow-GPU\\Git_Repo\\Gesture_Detector\\Image_Data\\train\\' + cat + '\\'
img_name = cat + "{}.png"
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        temp_image = cv2.imread(path + img_name.format(img_counter))
        # frame = cv2.resize(frame, (128, 128))
        # cv2.imwrite(path + img_name.format(img_counter), frame)
        # print("{} written!".format(img_name.format(img_counter)))
        # img_counter += 1
        if type(temp_image) == type(None):
            frame = cv2.resize(frame, (128, 128))
            cv2.imwrite(path + img_name.format(img_counter), frame)
            print("{} written!".format(img_name.format(img_counter)))
            img_counter += 1
        else:
            print(type(temp_image))
            while not isinstance(temp_image, type(None)):
                img_counter += 1
                print('Recalc: ' + str(img_counter))
                temp_image = cv2.imread(path + img_name.format(img_counter))
cam.release()
cv2.destroyAllWindows()
