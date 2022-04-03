# Saurav Panchal 
# 3 April 2022
# Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

# import necessary modules
import numpy as np
import pandas as pd
import cv2

import os

import pickle

from keras.preprocessing import image
from sklearn.metrics import accuracy_score

# defining / initializing variables
input_size = (512, 512)
y_test = []
y_pred = []
threshold = 435

def get_variance_of_laplacian(image):
	# returns the focus measure of image (variance of laplacian)
	return cv2.Laplacian(image, cv2.CV_64F).var()

digital_blur = pd.read_excel("C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\DigitalBlurSet.xlsx")
natural_blur = pd.read_excel("C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\NaturalBlurSet.xlsx")

digital_blur['MyDigital Blur'] = digital_blur['MyDigital Blur'].apply(lambda x : x.strip())
digital_blur = digital_blur.rename(index=str, columns={"Unnamed: 1": "Blur Label"})
natural_blur['Image Name'] = natural_blur['Image Name'].apply(lambda x : x.strip())

dir = "C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\DigitalBlurSet\\"

# load images
for file in os.listdir(dir):
    if file != '.DS_Store':
        image_path = dir + file
        img = image.load_img(image_path, target_size= input_size)
        blur = digital_blur[digital_blur['MyDigital Blur'] == file].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        var = get_variance_of_laplacian(gray)
        if var < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(file, "=> NADA")

print("--- Evaluated/Processed => ", dir, " ---")

dir = "C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\NaturalBlurSet\\"

# load images
for file in os.listdir(dir):
    if file != '.DS_Store':
        image_path = dir + file
        img = image.load_img(image_path, target_size=input_size)
        blur = natural_blur[natural_blur['Image Name'] == file.split('.')[0]].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        var = get_variance_of_laplacian(gray)
        if var < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(file, "=> NADA")

print("--- Evaluated/Processed => ", dir, " ---")

# load pickle file (y_test)
with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)

# accuracy 
print("Accuracy => ", (accuracy_score(y_test, y_pred)) * 100)
