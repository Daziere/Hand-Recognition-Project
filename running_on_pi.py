#!/usr/bin/env python
# coding: utf-8

# # Running the model on the raspberry pi

# All the imports needed to run the raspberry successfully. The version of tensor flow used in this was tensorflow 2, and opencv 4.6.
# To get Tensorflow and open CV I used the link below:
# https://www.youtube.com/watch?v=vekblEk6UPc&t=872s

# In[ ]:


import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2gray
import cv2
import imutils
import time
from time import sleep
from imutils.video import VideoStream
import tensorflow as tf
from tensorflow import keras
from sense_hat import SenseHat
from picamera import PiCamera


# def read_image(file_path) reads image file at file_path, cv2.imread loads the image in file_path, and then it gets returned downsized.

# In[ ]:


def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (28, 28),interpolation=cv2.INTER_CUBIC)


# Initializes the camera.

# In[ ]:


camera = PiCamera() #initalizes camera


# This sets up the sense hat for use. For more information use the link below:
# https://forums.raspberrypi.com/viewtopic.php?t=293163
# 

# In[ ]:


sense = SenseHat()
sense.set_rotation(180)
sense.low_light = True


# This loads the saved model.

# In[ ]:


model= keras.models.load_model("final_trained_model")


# This is code that infinitly runs the pi camera and processes the information from the pi camera.
# If more information is needed on camera function:
# https://picamera.readthedocs.io/en/release-1.13/api_camera.html
# 

# In[ ]:


while True:
    
    sense.clear()
    
    camera.start_preview()#starts the camera preview as a delay
    sleep(3)
    
    camera.stop_preview()#stops the preview shown
    camera.capture('/home/pi/project/image.jpg')   
    
    img = read_image('/home/pi/project/image.jpg')
    
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#coverts color spaces
    gray_img = cv2.resize(gray_img, (28, 28))
    gray_img = cv2.bitwise_not(gray_img)#inverts the array elements

    X_img = gray_img.reshape(1, 28, 28, 1)/255
    pred = model.predict(X_img)#runs the prediction model through the model made
    smpl_pred = np.argmax(pred, axis=-1)
    temp = pred[0, int(smpl_pred)]*100
    conf = str(int(temp))
    conf = conf + '%'

    num = str(smpl_pred)
    sense.show_message(num, text_colour=(0,0,255))#sets the color
    sleep(3)
    sense.show_message(conf, text_colour=(0,0,255))
    sense.clear()
    
    print(num)#prints the number on the display incase its missed from when it shows on the sense hat
    input("next")

