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

# ===== Plan =====
"""
1. Train model using data collected
2. Create a way for camera to take a picture 30 times each second
3. if 2 actions are found within 2 seconds execute command
"""


# cam = cv2.VideoCapture(0)
# cv2.namedWindow("test")
# model = load_model("Model_v2.7.h5")
# classes = {0: "Closed_Fist", 1: "Open_Hand", 2: "Other",
#            3: "Swipe_Left", 4: "Swipe_Right"}
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)
#
#     k = cv2.waitKey(1)
#     if k % 256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k % 256 == 32:
#         # SPACE pressed
#         frame = cv2.resize(frame, (128, 128))
#         frame = image.img_to_array(frame)
#         frame = np.expand_dims(frame, axis=0)
#         print(classes[model.predict_classes(frame)[0]])

def get_prediction(temp_model, temp_frame):
    global classes
    temp_frame = process_frame(temp_frame)
    return classes[temp_model.predict_classes(temp_frame)[0]]


def get_number_prediction(temp_model, temp_frame):
    temp_frame = process_frame(temp_frame)
    return temp_model.predict_classes(temp_frame)[0]


def collect_picture(camera, pred_model):
    result = []
    while len(result) < 50:
        temp, temp_frame = camera.read()
        temp_frame = process_frame(temp_frame)
        result.append(get_number_prediction(pred_model, temp_frame))
        print(len(result))
    return result


def getaction(actions):
    result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for action in actions:
        if action != 2:
            result[action] += 1
    values = list(result.values())
    values.sort()
    max1, max2 = values[-1], values[-2]
    final = []
    for key in result:
        if result[key] == max1 or result[key] == max2:
            final.append(key)
    return final


def process_frame(frame):
    if frame.shape != (1, 128, 128, 3):
        frame = cv2.resize(frame, (128, 128))
        frame = image.img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)
    return frame


def show_camera(cam):
    while True:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    model = load_model("Model_v1.15.h5")
    classes = {0: "Closed_Fist", 1: "Open_Hand", 2: "Other",
               3: "Swipe_Left", 4: "Swipe_Right"}
    # thread = Thread(target=show_camera(cam))
    # thread.start()
    # print("started")
    while True:
        ret1, frame1 = cam.read()
        frame1 = process_frame(frame1)
        time.sleep(0.5)
        ret2, frame2 = cam.read()
        frame2 = process_frame(frame2)
        if get_prediction(model, frame1) != "Other" and \
                get_prediction(model, frame2) != "Other":
            actions = getaction(collect_picture(cam, model))
            print(actions)

        time.sleep(0.1)
        print(get_prediction(model, frame1))
