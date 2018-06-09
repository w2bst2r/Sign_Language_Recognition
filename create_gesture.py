import cv2,os
import numpy as np
import pickle, sqlite3
from matplotlib import pyplot as plt

image_x, image_y = 50, 50
#%%
def create_folder(folder_name):
   if not os.path.exists(folder_name):
      os.mkdir(folder_name)
#%%
def init_create_folder_ifNotExist():
   if not os.path.exists("gestures"):
      os.mkdir("gestures")
#%%
def create_empty_images(folder_name, n_images):
   create_folder("gestures/"+folder_name)
   black = np.zeros(shape=(image_x, image_y, 1), dtype=np.uint8)
   for i in range(n_images):
      cv2.imwrite("gestures/"+folder_name+"/"+str(i+1)+".jpg", black)

#%%
def store_images(gesture_id):
   total_pics = 1200
   #firstly we create the class that contains black images
   if(gesture_id == str(0)):
      create_empty_images("0", total_pics)
      return
   cam = cv2.VideoCapture(1)
   x, y, w, h = 10, 40, 300, 300
   create_folder("gestures/"+str(gesture_id))
   picture_no = 0 # track the picture number that we're  in
   flag_start_capturing = False
   frames = 0

   while True:
      img = cam.read()[1]
      img = cv2.flip(img, 1)
      imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(imgGray,(5,5),0)
      blur = cv2.medianBlur(blur,1)
      thresh = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
      thresh = thresh[y:y+h, x:x+w]   # take the thresh bounded by the rectangle
#in countours, object to be found should be white and background should be black.
#arguments:first one is source image, second is contour retrieval mode, third is contour approximation method
      if frames > 50:
         picture_no += 1
         save_img = thresh# save the image in the rectangle part
         save_img = cv2.resize(save_img, (image_x, image_y))
         cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255)) #print capturing
         cv2.imwrite("gestures/"+str(gesture_id)+"/"+str(picture_no)+".jpg", save_img)

      cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
      cv2.putText(img, str(picture_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))  #print picture number
      cv2.imshow("Capturing gesture", img)
      cv2.imshow("thresh", thresh)
      keypress = cv2.waitKey(1)
      if keypress == ord('c'):
         if flag_start_capturing == False:
            flag_start_capturing = True
         else:
            flag_start_capturing = False
            frames = 0
      if flag_start_capturing == True:
         frames += 1
      if picture_no == total_pics:
         break
      #if q is pressed quit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   cam.release()
   cv2.destroyAllWindows()
#%%
init_create_folder_ifNotExist()
gesture_id = input("Enter gesture no.: ")
gesture_name = input("Enter gesture name/text: ")
store_images(gesture_name)




