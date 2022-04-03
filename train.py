# Saurav Panchal 
# 3 April 2022
# Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

# importing necessary libraries
import numpy as np
import pandas as pd

import os

import pickle

from keras.preprocessing import image

X_train = list()
y_train = list()

input_size = (192, 192)

artificial_dir = "C:\\Users\\saura\\Documents\\Datasets\\CERTH_ImageBlurDataset\\TrainingSet\\Artificially-Blurred\\"
natural_dir = "C:\\Users\\saura\\Documents\\Datasets\\CERTH_ImageBlurDataset\\TrainingSet\\Naturally-Blurred\\"
undistorted_dir = "C:\\Users\\saura\\Documents\\Datasets\\CERTH_ImageBlurDataset\\TrainingSet\\Undistorted\\"

dirs = list((artificial_dir, natural_dir, undistorted_dir))

for dir in dirs:
    for file in os.listdir(dir):
        if file != ".DS_Store":
            img_path = dir + file
            img = image.load_img(img_path, target_size = input_size)
            X_train.append(np.asarray(img))
            y_train.append(1)
        else:
            print(file, "=> NADA")
    print("--- Loaded => ", dir, " ---")
print("--- Trainset Loaded ---")

# Create picklefile for X_train
with open("X_train.pkl", "wb") as picklefile:
    pickle.dump(X_train, picklefile)

# Create picklefile for y_train
with open("y_train.pkl", "wb") as picklefile:
    pickle.dump(y_train, picklefile)