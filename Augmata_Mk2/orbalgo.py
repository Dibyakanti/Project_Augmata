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
    #For first frame
    if skip:
        old = curr
        skip = False
    #Detect keypoints
    keypCurr, desCurr = orb.detectAndCompute(curr, None)
    keypOld, desOld = orb.detectAndCompute(old, None)
    #Match keypoints
    matches = sorted(bfMatcher.match(desCurr, desOld), key = lambda x:x.distance)
    #Select out the matching keypoint locations
    currLocs = []
    oldLocs = []
    for i, m in enumerate(matches[:5]):
        currLocs.append(keypCurr[m.queryIdx].pt)
        oldLocs.append(keypOld[m.queryIdx].pt)
    #Draw line from feature at n-1 to n
    for kp in range(len(currLocs)-1):
        cv2.line(curr, (int(oldLocs[kp][0]), int(oldLocs[kp][1])),
                (int(currLocs[kp][0]), int(currLocs[kp][1])), (255,0,0))
    #Draw matching keypoints
    orbImage = drawKeyPoints(curr.copy(), keypCurr, (0,255,0))
    oldImage = drawKeyPoints(old.copy(), keypOld, (0,255,0))
    final = np.append(orbImage, oldImage, axis = 1)
    cv2.imshow("Image", final)
    #

    matching = 0
    matching = cv2.drawMatches(curr, keypCurr, old, keypOld, matches[:10], matching, flags = 2)
    cv2.imshow("matching", matching)
    cv2.imshow("Show", curr)
    if cv2.waitKey(50) == ord('q'): break
    old = curr

cap.release()
cv2.destroyAllWindows()
