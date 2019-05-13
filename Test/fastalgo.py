import cv2
import numpy as np
from sklearn.cluster import KMeans
cap = cv2.VideoCapture(1)

fast = cv2.FastFeatureDetector_create()

def drawKeyPoints(frame, keypoints, color):
    for point in keypoints:
        out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color, 1)
    return out


while True:
    _, frame = cap.read()
    keypoints = fast.detect(frame, None)
    non_max_sup = drawKeyPoints(frame.copy(), keypoints, (0,0,255))
    cv2.imshow("Non_Max_Suppression", non_max_sup)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
