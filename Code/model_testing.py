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
from tensorflow.keras.preprocessing import image
import cv2
from threading import Thread
import time



cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
model = load_model("Model_v2.8.h5")
classes = {0: "Closed_Fist", 1: "Open_Hand", 2: "Other",
           3: "Swipe_Left", 4: "Swipe_Right"}
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
        frame = cv2.resize(frame, (128, 128))
        frame = image.img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)
        print(classes[model.predict_classes(frame)[0]])
