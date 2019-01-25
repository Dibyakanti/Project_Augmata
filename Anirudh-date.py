import cv2
import numpy as np
import datetime
#setting backround image for the date
background_img=cv2.imread('/home/anirudh/Downloads/black.jpg')
#obtaining date
date = datetime.datetime.now()
#making date a string
str_date= str(date.year) + "-" + str(date.month) + "-" + str(date.day)
#adding string to the backround
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
cv2.putText(background_img, str_date, (15,115), font , 1, (255,0,0),3, cv2.LINE_AA)
cap = cv2.VideoCapture(0)
while(1):
     _, frame = cap.read()
     rows,cols,channels = background_img.shape
     #creating a translucent image of date over the video
     modification = cv2.addWeighted(background_img , 0.4 , frame[0:rows , 0:cols], 0.6 , 0)
     frame[0:rows, 0:cols ] = modification
     cv2.imshow('res',frame)
     k = cv2.waitKey(30) & 0xFF 
     if k == 27 :
         break  
cv2.destroyAllWindows()
