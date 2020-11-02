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

image_path = 'D:\\Users\\Master\\Documents\\A.I Project\\TensorFlow' \
             '\\TensorFlow-GPU\\Git_Repo\\Gesture_Detector\\Image_Data\\'
train_path = image_path + 'train\\'
test_path = image_path + 'test\\'
image_shape = (128, 128, 3)

image_gen = ImageDataGenerator(fill_mode='nearest')

model = Sequential()
# model.add(Conv2D(filters=256, kernel_size=(8, 8), input_shape=image_shape,
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Conv2D(filters=256, kernel_size=(8, 8), input_shape=image_shape,
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Conv2D(filters=128, kernel_size=(8, 8), input_shape=image_shape,
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(4, 4)))
#
model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape,
                 activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape,
                 activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape,
                 activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape,
                 activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

early_stop = EarlyStopping(monitor='val_accuracy', patience=2)

# opt = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='Adam',
              metrics=['accuracy'])
train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=8,
                                                class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               class_mode='categorical',
                                               batch_size=16,
                                               shuffle=False)
response = input('Run Model Y/N')
if response == 'Y':
    results = model.fit_generator(train_image_gen, epochs=50,
                                  validation_data=test_image_gen,
                                  callbacks=[early_stop])
    model.save('Model_v1.17.h5')
    predictions = model.predict_classes(test_image_gen)
    print(classification_report(test_image_gen.classes, predictions))
# os.system("shutdown /s /t 1")
