import cv2
import numpy as np
from sklearn.cluster import KMeans
cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create()

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
    keypoints = fast.detect(frame, None)
    non_max_sup, frame_keypoints = drawKeyPoints(frame.copy(), keypoints, (0,0,255))
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(frame_keypoints)

    emp = np.zeros(frame.shape)
    emp = drawKMeans(emp.copy(), keypoints, kmeans.labels_, (255,255,255), (255,255,0))
    cv2.imshow("Emp", emp)
    cv2.imshow("Non_Max_Suppression", non_max_sup)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
