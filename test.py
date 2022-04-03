# Saurav Panchal 
# 3 April 2022
# Image Blur Detection On CERTH_ImageBlurDetection Dataset - @Task by CloudSEK (Karnataka)

# importing necessary libraries
import numpy as np
import pandas as pd

import os

import pickle

from keras.preprocessing import image

input_size = (192, 192)
flag = 0

X_test = []
y_test = []

digital_blur = pd.read_excel("C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\DigitalBlurSet.xlsx")
natural_blur = pd.read_excel("C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\NaturalBlurSet.xlsx")

digital_blur["MyDigital Blur"] = digital_blur["MyDigital Blur"].apply(lambda x : x.strip())
digital_blur = digital_blur.rename(index = str, columns = {"Unnamed: 1": "Blur Label"})

natural_blur["Image Name"] = natural_blur["Image Name"].apply(lambda x: x.strip())

digital_blur_dir = "C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\DigitalBlurSet\\"
natural_blur_dir = "C:\\Users\\saura\\Documents\\Datasets\CERTH_ImageBlurDataset\\EvaluationSet\\NaturalBlurSet\\"

dirs = list((digital_blur_dir, natural_blur_dir))

for dir in dirs:
    if dir == digital_blur_dir:
        flag = 1
    else:
        flag = 0

    for file in os.listdir(dir):
        if file != ".DS_Store":
            img_path = dir + file
            img = image.load_img(img_path, target_size = input_size)
            X_test.append((1/255) * np.asarray(img))
            # print(natural_blur["Image Name"], "\n", file.split(".")[0])
            # print(natural_blur["Image Name"])
            if flag == 0:
                blur = natural_blur[natural_blur["Image Name"] == file.split(".")[0]].iloc[0]['Blur Label'] 
            elif flag == 1:
                blur = digital_blur[digital_blur["MyDigital Blur"] == file].iloc[0]['Blur Label']
            
            if blur == 1:
                y_test.append(1)
            else:
                y_test.append(0)
        else:
            print(file, "=> NADA")
    print("--- Loaded => ", dir, " ---")
print("--- Testset Loaded ---")

with open("X_test.pkl", "wb") as picklefile:
    pickle.dump(X_test, picklefile)

with open("y_test.pkl", "wb") as picklefile:
    pickle.dump(y_test, picklefile)