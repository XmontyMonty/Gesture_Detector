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
import ctypes

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
    while len(result) < 25:
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


def get_single_action(cam, model):
    results = []
    action_counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    while len(results) < 15:
        ret, frame = cam.read()
        frame = process_frame(frame)
        results.append(get_number_prediction(model, frame))
    for action in results:
        action_counter[action] += 1
    max_val = max(list(action_counter.values()))
    for key in action_counter:
        if action_counter[key] == max_val:
            return key


def switch_windows_right():
    user32 = ctypes.windll.user32
    user32.keybd_event(0x5B, 0, 0, 0)
    user32.keybd_event(0xA2, 0, 0, 0)
    user32.keybd_event(0x27, 0, 0, 0)
    user32.keybd_event(0x5B, 0, 2, 0)
    user32.keybd_event(0xA2, 0, 2, 0)
    user32.keybd_event(0x27, 0, 2, 0)


def switch_windows_left():
    user32 = ctypes.windll.user32
    user32.keybd_event(0x5B, 0, 0, 0)
    user32.keybd_event(0xA2, 0, 0, 0)
    user32.keybd_event(0x25, 0, 0, 0)
    user32.keybd_event(0x5B, 0, 2, 0)
    user32.keybd_event(0xA2, 0, 2, 0)
    user32.keybd_event(0x25, 0, 2, 0)


def close_windows():
    user32 = ctypes.windll.user32
    user32.keybd_event(0x5B, 0, 0, 0)
    user32.keybd_event(0x44, 0, 0, 0)
    user32.keybd_event(0x25, 0, 2, 0)
    user32.keybd_event(0x5B, 0, 2, 0)


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    model = load_model("Model_v2.9.h5")
    classes = {0: "Closed_Fist", 1: "Open_Hand", 2: "Other",
               3: "Swipe_Left", 4: "Swipe_Right"}
    # thread = Thread(target=show_camera(cam))
    # thread.start()
    # print("started")
    while True:
        action1 = get_single_action(cam, model)
        if action1 == 1:
            action2 = get_single_action(cam, model)
            action3 = get_single_action(cam, model)
            if action2 == 3 or action3 == 3:
                switch_windows_right()
                print("Open Hand ==> Swipe Left")
            elif action2 == 4 or action3 == 4:
                switch_windows_left()
                print("Open Hand ==> Swipe Right")
            elif action2 == 0 or action3 == 0:
                close_windows()
                print("Open Hand ==> Closed Fist")
        print(classes[action1])
        time.sleep(0.005)
