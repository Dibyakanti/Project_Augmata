import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()
skip = True
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

def drawKeyPoints(frame, keypoints, color):
    for point in keypoints:
        out = cv2.circle(frame, tuple(int(i) for i in point.pt), 3, color, 1)
    return out

def drawTracking(prev, current):
    return 0

while True:
    _, curr = cap.read()
    if skip:
        old = curr
        skip = False
    keypCurr, desCurr = orb.detectAndCompute(curr, None)
    keypOld, desOld = orb.detectAndCompute(old, None)
    matches = sorted(bfMatcher.match(desCurr, desOld), key = lambda x:x.distance)
    # orbImage = drawKeyPoints(curr.copy(), keypCurr, (0,255,0))
    # oldImage = drawKeyPoints(old.copy(), keypOld, (0,255,0))
    # final = np.append(orbImage, oldImage, axis = 1)
    matching = 0
    matching = cv2.drawMatches(curr, keypCurr, old, keypOld, matches[:10], matching, flags = 2)
    cv2.imshow("matching", matching)
    # cv2.imshow("Image", final)
    if cv2.waitKey(1) == ord('q'): break
    old = curr

cap.release()
cv2.destroyAllWindows()
