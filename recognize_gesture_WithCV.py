import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model
import matplotlib.pyplot as plt

ModelPathCV = 'karim_model_withCV_V1.h5'
#%%map number to letter
# class_indices: (alphabet_in_Number <-->classNumber)
def map_class(num):
   if num ==0: return 'None'
   if num ==1: return 'A'
   if num ==2: return 'B'
   if num ==3: return 'C'
   if num ==4: return 'D'
   if num ==5: return 'E'
   if num ==6: return 'F'
   if num ==7: return 'G'
   if num ==8: return 'H'
   if num ==9: return 'I'
   if num ==10: return 'J'
   if num ==11: return 'K'
   if num ==12: return 'L'
   if num ==13: return 'M'
   if num ==14: return 'N'
   if num ==15: return 'O'
   if num ==16: return 'P'
   if num ==17: return 'Q'
   if num ==18: return 'R'
   if num ==19: return 'S'
   if num ==20: return 'T'
   if num ==21: return 'U'
   if num ==22: return 'V'
   if num ==23: return 'W'
   if num ==24: return 'X'
   if num ==25: return 'Y'
   if num ==26: return 'Z'
#%%
def keras_process_image(img):
   img = cv2.resize(img, (50, 50))#img,50,50 in our case
   #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
   img = np.array(img, dtype=np.float32) #transform it to array
   #(img(#images,width,height,#channels))
   img = np.reshape(img, (1, 50, 50, 1))
   return img
#%%
def keras_predict(model, image):
   processed = keras_process_image(image)
   pred_probab = model.predict(processed)[0]#return probabilities of each class
   pred_class = list(pred_probab).index(max(pred_probab))#return the index of max
   return pred_probab, pred_class
#%%global variables
prediction = None
model = load_model(ModelPathCV)
x, y, w, h = 10, 40, 300, 300
letter = None
#%%) # the third last arguments are for text color., y specify the position of the text
#def put_splitted_text_in_blackboard(blackboard, splitted_text):
#   y = 200
#  # for text in splitted_text:
#      cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (100, 255, 100))
#      y += 50
#%%
def put_text_in_blackboard(blackboard,letter,x,y,font):
   cv2.putText(blackboard, letter, (x, y), font, 2, (100, 255, 100))
#%%  
def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
#%%
cam = cv2.VideoCapture(1)

delay = 80
while True:
   #print(delay)
   plt.close()
   fig = plt.figure(figsize=(10,6))
   delay+=1
   img = cam.read()[1]
   img = cv2.flip(img, 1)
   imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(imgGray,(5,5),0)
   blur = cv2.medianBlur(blur,1)
   cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) #draw the rectangle
   #thresh = cv2.threshold(blur,160,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] 
   thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
         cv2.THRESH_BINARY_INV,255,15)
   thresh = thresh[y:y+h, x:x+w]
   
   pred_probab, pred_class = keras_predict(model, thresh)
   max_pred_probab = max(pred_probab)
   pred_probab = pred_probab*100
  
   # when predicting image is resized and preprocesed
   cv2.imshow("img", img)
   cv2.imshow("thresh", thresh)
   blackboard = np.zeros((280, 460, 3), dtype=np.uint8) #second argument defines the width of the blackboard. first aqrgument is the height and it is fixed(480).
   
   #figure plotting
   if(delay % 20 == 0):
    #  pred_probab, pred_class = keras_predict(model, thresh)      
      delay = 80

      y_hist = [pred_probab[0],pred_probab[1],pred_probab[2],pred_probab[3],pred_probab[4],pred_probab[5],pred_probab[6],pred_probab[7],pred_probab[8],pred_probab[9],pred_probab[10],pred_probab[11],pred_probab[12],pred_probab[13],pred_probab[14],pred_probab[15],pred_probab[16],pred_probab[17],pred_probab[18],pred_probab[19],pred_probab[20],pred_probab[21],pred_probab[22],pred_probab[23],pred_probab[24],pred_probab[25],pred_probab[26]]
      
      N = len(y_hist)
      x_hist = range(N)
      width = 1/1.1
      plt.xticks(x_hist, ('None','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'))
      plt.tick_params(axis='both', which='major', labelsize=20)
      plt.bar(x_hist, y_hist, width,align='center', alpha=0.5, color="red")  
      #this is for showing directly the histogram
      hist_to_array = fig2rgb_array(fig)
      cv2.imshow("histogram", hist_to_array)
#      fig.savefig('histogram.png')
#      histogram = cv2.imread('histogram.png',1) 
#      cv2.imshow("histogram", histogram)
      
   if max_pred_probab*100 > 85 and pred_class != 0:
      classOfletter = pred_class #predicted class number
      letter= map_class(classOfletter) #mapping to the letter
      #print(pred_class, max_pred_probab*100,letter)
      put_text_in_blackboard(blackboard, letter,4,70,cv2.FONT_HERSHEY_TRIPLEX)
      put_text_in_blackboard(blackboard, "Probability :" +str(round(max_pred_probab*100,2)),4,200,cv2.FONT_HERSHEY_PLAIN)
   else:
      letter='None'
      put_text_in_blackboard(blackboard, letter,4,70,cv2.FONT_HERSHEY_TRIPLEX)
   #if cv2.waitKey(1) == ord('s'):
      #x += 30
   #res = np.hstack((img, blackboard)) #merging the img and blackboard
   cv2.imshow("blackboard", blackboard)


   if cv2.waitKey(1)  == ord('q'):
      break
plt.close()
cam.release()
cv2.destroyAllWindows()