# # Saurav Panchal 
# # 3 April 2022
# # Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

import numpy as np
import pickle

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

input_size = (96, 96)

with open("X_train.pkl", "rb") as picklefile:
    X_train = pickle.load(picklefile)

with open("y_train.pkl", "rb") as picklefile:
    y_train = pickle.load(picklefile)

with open("X_test.pkl", "rb") as picklefile:
    X_test = pickle.load(picklefile)

with open("y_test.pkl", "rb") as picklefile:
    y_test = pickle.load(picklefile)


# model = Sequential()


# model.add(Convolution2D(64, (5, 5), activation = "relu", input_shape = (input_size[0], input_size[1], 3)))
# model.add(MaxPooling2D(pool_size = (2, 2)))


# # model.add(Convolution2D(64, (5, 5)))
# # model.add(Activation("relu"))
# # model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Flatten())


# # model.add(Dense(1024))
# # model.add(Activation("relu"))
# # model.add(Dropout(0.8))


# # model.add(Dense(512))
# # model.add(Activation("relu"))
# # model.add(Dropout(0.5))

# # model.add(Dense(256))
# # model.add(Activation("relu"))
# # model.add(Dropout(0.5))


# model.add(Dense(2))
# model.add(Activation("softmax"))
# # sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0.0, nesterov = False)
# model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# train = np.stack(X_train)
# test = np.stack(X_test)
# train_label = to_categorical(y_train)
# test_label = to_categorical(y_test)

# model.fit(train, train_label, batch_size = 128, epochs = 2, verbose = 1)
# print("--- Model Training Completed ---")

# (loss, train_accuracy) = model.evaluate(train, train_label, batch_size = 128, verbose = 1)
# (loss, test_accuracy) = model.evaluate(test, test_label, batch_size = 128, verbose = 1)

# print("--- Accuracy ---")
# print("Train => ", train_accuracy)
# print("Test => ", test_accuracy)



# My LeNet architecture
model = Sequential()

# conv filters of 5x5 each

# Layer 1
model.add(Convolution2D(32, 5, 5, input_shape=(192, 192, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(32, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))    

model.add(Dropout(0.25))

# Layer 3
model.add(Convolution2D(64, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))    


model.add(Flatten())

# Layer 4
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))


# Layer 5
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Layer 6
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Normalizing images
X_stacked = np.stack(X_train)
X_norm = X_stacked/255.

#Converting labels to categorical values
y_cat = to_categorical(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_cat, test_size=0.2, random_state=42)

# Data augmenter
dg = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

model.fit_generator(dg.flow(X_train, y_train), samples_per_epoch=3000, nb_epoch=10, validation_data=dg.flow(X_test, y_test), nb_val_samples=300)

# model.save_weights('model_weights.h5')