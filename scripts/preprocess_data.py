import os 
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import random
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report,confusion_matrix
     


def view_image(target_dir, target_class):
    # Properly join paths
    target_folder = os.path.join(target_dir, target_class)
    
    # Check if the directory exists
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"The directory {target_folder} does not exist.")
    
    # Randomly select an image
    random_image = random.sample(os.listdir(target_folder), 1)[0]
    
    # Load and display the image
    img_path = os.path.join(target_folder, random_image)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    
    return img

img = view_image("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train", "Non Mask")
img = view_image("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train","Mask")

data=[]
labels=[]
no_mask=os.listdir("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Non Mask\\")
for a in no_mask:

    image = cv2.imread("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Non Mask\\"+a,)
    image = cv2.resize(image, (224, 224))


    data.append(image)
    labels.append(0)

no_mask=os.listdir("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Non Mask\\")
for a in no_mask:

    image = cv2.imread("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Non Mask\\"+a,)
    image = cv2.resize(image, (224, 224))


    data.append(image)
    labels.append(0)


mask=os.listdir("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Mask\\")
for a in mask:

    image = cv2.imread("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Mask\\"+a,)
    image = cv2.resize(image, (224, 224))


    data.append(image)
    labels.append(1)


mask=os.listdir("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Mask\\")
for a in mask:

    image = cv2.imread("C:\\Users\\moham\\Desktop\\mask-detection\\data\\New_Masks_Dataset\\Train\\Mask\\"+a,)
    image = cv2.resize(image, (224, 224))


    data.append(image)
    labels.append(1)


data = np.array(data) / 255.0
labels = np.array(labels)