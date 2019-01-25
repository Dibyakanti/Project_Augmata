import cv2
import numpy as np
import datetime
date = datetime.datetime.now()
#creating paths for day and month icons
date_path = "/home/anirudh/icons/" + str(date.day) +".png" 
month_path = "/home/anirudh/icons/" + str(date.month) + "month.png"
#acessing day and month icons
date_img = cv2.imread(date_path) 
month_img = cv2.imread(month_path)
cap = cv2.VideoCapture(0)
while(1):
     _, frame = cap.read()
     rows,cols,channels = date_img.shape
     #creating a translucent image of day over the video
     date_modification = cv2.addWeighted(date_img , 0.3 , frame[0:rows , 0:cols], 0.7 , 0)
     frame[0:rows, 0:cols ] = date_modification
     #creating a translucent image of month over the video
     month_modification = cv2.addWeighted(month_img , 0.3 , frame[0:rows , cols:2*cols], 0.7 , 0)
     frame[0:rows, cols:2*cols ] = month_modification
     cv2.imshow('res',frame)
     k = cv2.waitKey(30) & 0xFF 
     if k == 27 :
         break  
cv2.destroyAllWindows()
