import cv2
import numpy as np
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()

def drawKeyPoints(frame, keypoints, color):
    frame_keypoints = []
    for point in keypoints:
        frame_keypoints.append(np.asarray(point.pt))
        out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color, 1)
    return out, frame_keypoints

def drawKMeans(frame, keypoints, labels, color1, color2):
    for (i, point) in enumerate(keypoints):
        if labels[i] == 0:
            out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color1, 1)
        if labels[i] == 1:
            out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color2, 1)
    return out

while True:
    _, frame = cap.read()
    keyp = orb.detect(frame, None)
    keyp, des = orb.compute(frame, keyp)
    orbimage, frame_keypoints = drawKeyPoints(frame.copy(), keyp, (0,255,0))
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(frame_keypoints)
    emp = np.zeros(frame.shape)
    emp = drawKMeans(emp.copy(), keyp, kmeans.labels_, (255,255,255), (255,255,0))
    cv2.imshow("Orb", orbimage)
    cv2.imshow("Emp", emp)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
