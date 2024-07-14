import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
image_directory = 'datasets/'
datasets = []
labels = []
INPUT_SIZE = 64

# Load 'no' tumor images
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
for image_name in no_tumor_images:
    if image_name.endswith(".jpg"):
        image_path = os.path.join(image_directory, "no", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            datasets.append(image)
            labels.append(0)

# Load 'yes' tumor images
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))
for image_name in yes_tumor_images:
    if image_name.endswith(".jpg"):
        image_path = os.path.join(image_directory, "yes", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            datasets.append(image)
            labels.append(1)

datasets = np.array(datasets)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)
model.save("brainTumorDetectionCategorical.h5")
