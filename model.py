# Saurav Panchal 
# 3 April 2022
# Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

import numpy as np
import pandas as pd
import os
import pickle

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Flatten, Dense, Activation, Dropout

from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

input_size = (96, 96)

with open("X_train.pkl", "rb") as picklefile:
    X_train = pickle.load(picklefile)

with open("y_train.pkl", "rb") as picklefile:
    y_train = pickle.load(picklefile)

with open("X_test.pkl", "rb") as picklefile:
    X_test = pickle.load(picklefile)

with open("y_test.pkl", "rb") as picklefile:
    y_test = pickle.load(picklefile)

