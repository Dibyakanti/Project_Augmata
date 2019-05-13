import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)
orb = cv2.ORB_create()

def drawKeyPoints(frame, keypoints, color):
    for point in keypoints:
        out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color, 1)
    return out


while True:
    _, frame = cap.read()
    keyp = orb.detect(frame, None)
    keyp, des = orb.compute(frame, keyp)
    orbimage = drawKeyPoints(frame.copy(), keyp, (0,255,0))
    cv2.imshow("Orb", orbimage)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
