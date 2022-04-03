# Saurav Panchal 
# 3 April 2022
# Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

# import necessary modules
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation

from keras.utils.np_utils import to_categorical

# initializing / defining variables
input_size = (96, 96)

# loading pickle files
with open("X_train.pkl", "rb") as picklefile:
    X_train = pickle.load(picklefile)

with open("y_train.pkl", "rb") as picklefile:
    y_train = pickle.load(picklefile)

with open("X_test.pkl", "rb") as picklefile:
    X_test = pickle.load(picklefile)

with open("y_test.pkl", "rb") as picklefile:
    y_test = pickle.load(picklefile)

# creating simple Sequential model
model = Sequential()

# adding layers to Sequential model
model.add(Convolution2D(64, (5, 5), activation = "relu", input_shape = (input_size[0], input_size[1], 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

train = np.stack(X_train)
test = np.stack(X_test)
train_label = to_categorical(y_train)
test_label = to_categorical(y_test)

model.fit(train, train_label, batch_size = 128, epochs = 2, verbose = 1)
print("--- Model Training Completed ---")

# assigning accuracies to variables
(loss, train_accuracy) = model.evaluate(train, train_label, batch_size = 128, verbose = 1)
(loss, test_accuracy) = model.evaluate(test, test_label, batch_size = 128, verbose = 1)

# printing accuracies
print("--- Accuracy ---")
print("Train => ", train_accuracy)
print("Test => ", test_accuracy)