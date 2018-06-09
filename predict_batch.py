# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
NUM_SAMPLE = 32400
BATCH_SIZE =30
#this is to print pred_class in correct format by suppressing e-1...
np.set_printoptions(precision=3, suppress=True)

ModelPath = "karim_model_withCV.h5"
imagePath = r'C:\Users\Karim El Guermai\Desktop\PROGRAMMING\FinalProject_Last\gestures'
#%%
# create_img_generator is for data augmentation, it is optional
def create_img_generator():
    return ImageDataGenerator()    
    
def generateData(path):
    #If you're not using data augmentation, ImageDataGenerator().flow_from_direcotry()
    predict_generator = create_img_generator().flow_from_directory(
            imagePath,
            target_size = (50, 50),
            batch_size = BATCH_SIZE ,
            color_mode='grayscale',   #it is by default 'rgb'. 1 means grayscale
            #you set classes and class_mode to none if you don't have the label of the samples
           # classes = None,
           # class_mode=None, #if you have image classes mixed
            #save_to_dir # store augmented data to save_to_dir
            #classes = ['A','X','C','D','E','F','G','H'] ,
# ifyou don't specify it, classes names will be inferred form the directory structure],
            #  # keep data in the same order
            shuffle = False,
            seed = 42)   
    return predict_generator

#%%
model = load_model(ModelPath)
test_batches = generateData(ModelPath)
print(test_batches.class_indices)   # print the labels of each class

#test_imgs, labels = next(test_batches) no need as we grab all test_batches at once
predictions = model.predict_generator(test_batches, steps=NUM_SAMPLE/BATCH_SIZE) #STEPS= #of batches
#if you omit interpolation,il will not work
#plt.imshow(test_imgs[4], cmap = 'gray', interpolation = 'bicubic')
#%% # confusion matrix
y_pred = np.argmax(predictions, axis=1)  #get the max values of each row and print its index column
#y_pred = predictions>0.5
#target_names = 
#y_test = np.array([0]*200+[1]*200+[2]*200+[3]*200+[4]*200+[5]*200+[6]*200+[7]*200+[8]*200+[9]*200+[10]*200+[11]*200+[12]*200+[13]*200+[14]*200+[15]*200+[16]*200+[17]*200+[18]*200+[19]*200+[20]*200+[21]*200+[22]*200+[23]*200+[24]*200+[25]*200+[26]*200)
y_test = np.array([0]*1200+[1]*1200+[10]*1200+[11]*1200+[12]*1200+[13]*1200+[14]*1200+[15]*1200+[16]*1200+[17]*1200+[18]*1200+[19]*1200+[2]*1200+[20]*1200+[21]*1200+[22]*1200+[23]*1200+[24]*1200+[25]*1200+[26]*1200+[3]*1200+[4]*1200+[5]*1200+[6]*1200+[7]*1200+[8]*1200+[9]*1200)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)