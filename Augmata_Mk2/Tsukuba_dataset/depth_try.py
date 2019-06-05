import numpy as np
import cv2
from matplotlib import pyplot as plt

def tsukuba():
    imgL = cv2.imread('tsukuba_l.ppm',0)
    imgR = cv2.imread('tsukuba_r.ppm',0)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=19)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def stereo_cam():
    cam_l = cv2.VideoCapture(2)
    cam_r = cv2.VideoCapture(1)
    while True:
        _, imgL = cam_l.read()
        _, imgR = cam_r.read()
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
        disparity = stereo.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()
        out = np.append(imgL, imgR, axis = 1)
        cv2.imshow('out', out)
        if cv2.waitKey(1) == ord('q'): break

def file_read():
    imgL = cv2.imread('test2.jpg',0)
    imgR = cv2.imread('test1.jpg',0)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=19)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

if __name__ == "__main__":
    file_read()
