import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

lk_params = dict(winSize = (15,15), maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255, (100,3))

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

_, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old, mask = None, **feature_params)
mask = np.zeros_like(old)

while True:
    _, frame = cap.read()
    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old, curr, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    for i, (new,old1) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old1.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        curr = cv2.circle(curr, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(curr, mask)
    cv2.imshow("Show", img)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(10) == 27: break
    old = curr.copy()
    p0 = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()
