# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
#this is to print pred_class in correct format by suppressing e-1...
np.set_printoptions(precision=3, suppress=True)

ModelPath = 'karim_model.h5'
ModelPathCV ='karim_model_withCV.h5'
imagePath = r'C:\Users\Karim El Guermai\Desktop\PROGRAMMING\FinalProject_Last\gestures _splitten\validate\10\1100.jpg'
#%%
def get_image_size(imagePath):
    img = cv2.imread(imagePath, 0)
    return img.shape
#%%
#read and preprocess one image
def keras_process_image(imagePath):
    img = cv2.imread(imagePath, 0)
    img = cv2.resize(img, (50, 50))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, 50, 50, 1))   #(img,(#images,width,height,#channels))
    return img    #return preprocessed image as a numpy array
#%%
#predicting
def keras_Predict(model,img):
    #transfer the numpy array to list. the list contains the probabilities of each# class in order. 
    pred_probab  = model.predict(img)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return pred_probab, pred_class
#%% pred_probab contains the probability for all the classes
# this function the probability of an image to a specific class
def print_probability(pred_probab, classNumber):
    print(list(pred_probab*100)[classNumber])
#%%
image_width, image_height = get_image_size(imagePath)# get image shape
model = load_model(ModelPathCV) # load the model
preprocessedImage = keras_process_image(imagePath)
pred_probab, pred_class = keras_Predict(model, preprocessedImage)
#%%
#print(list(pred_probab)*100)
print(max(pred_probab))
print(pred_class)

#cm = confusion_matrix(test_labels, rounded_predictions)

