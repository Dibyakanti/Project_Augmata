#Robotics Club - IITG

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
import os
from objloader_simple import *

font = cv2.FONT_HERSHEY_COMPLEX

def make_marker(index):
    '''
    Function to create marker from default dict
    and return index marker
    Input   : index (int32)
    Return  : marker (numpy array)
    '''
    if index > 249:
        print('Exceeded dict size')
        return 0
    #Get predefined dictionary - 6x6 size and 250 markers
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    #drawMarker(dict, markerid, output size)
    marker = aruco.drawMarker(aruco_dict, index, 700)
    cv2.imshow('marker', marker)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
    cv2.imwrite('markers/m'+str(index)+'.jpg', marker)
    return marker

def detect_marker(show):
    '''
    Funtion to perform marker detection
    Input   : show - 0  : return values without display
                   - 1  : return values after displaying
    Return  : corners   : numpy array of corners of markers
              ids       : ids of markers
              rejected  : corners of rejected points
    '''
    cap = cv2.VideoCapture(1)
    while(1):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        #Get the params for the dictionary
        params = aruco.DetectorParameters_create()
        #Dectection
        #detectMarkers(inp, dict, corners, ids, params, rejetcs)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if show == 0:
            cap.release()
            return corners, ids, rejected, frame
        detected = aruco.drawDetectedMarkers(frame, corners)
        if np.all(ids != None):
            #print('Detected :',len(ids))
            for i in range(len(ids)):
                cv2.putText(detected, str(ids[i][0]),
                                    tuple(corners[i][0][2]),
                                    font, 0.5, (0, 0, 255), 1, 4)
        cv2.imshow('Detection', detected)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return corners, ids, rejected

def calib():
    '''
    By njanuridh - github - https://github.com/njanirudh/Aruco_Tracker/blob/master/
    Function to perform camera calibration
    Return  : mtx   : camera matrix
              dist  : distortion coefficients
              rvecs  : rotation vectors
              tvecs  : translation vectors
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('calib_images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                    imgpoints, gray.shape[::-1],None,None)
    return mtx, dist, rvecs, tvecs

def render(img, obj, projection, model, color=False):
    """
    By juang - github - https://github.com/juangallostra/augmented-reality
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    By juang - github - https://github.com/juangallostra/augmented-reality
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    #print(rot_1)
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def multi_marker_pose():
    '''
    Function to perform pose estimation on ArUco markers
    Return  : rvec_list : list of estimated rotation vectors
              tvec_list : list of estimated translation vectors
    '''
    cap = cv2.VideoCapture(1)
    print('Calibrating....')
    mtx, dist, _, _ = calib()
    print('Calibrated')
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'models/ge.obj'), swapyz=True)
    ref = cv2.imread(os.path.join(dir_name, 'reference/marker4.jpg'), 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters_create()
    corners_src, ids_src, rejected_src = aruco.detectMarkers(ref, aruco_dict,
                                            parameters = params)
    while(1):

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if np.all(ids != None):
            rvec_list = []
            tvec_list = []
            aruco.drawDetectedMarkers(frame, corners)
            for i in range(len(ids)):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, mtx, dist)
                #aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.05)
                cv2.putText(frame, str(ids[i][0]),
                                    tuple(corners[i][0][2]),
                                    font, 0.5, (0, 0, 255), 1, 4)
                rvec_list.append(rvecs)
                #print(tvecs[0][0])
                tvec_list.append(tvecs)

            homography, mask = cv2.findHomography(corners_src[0], corners[0], cv2.RANSAC, 5.0)
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(mtx, homography)
                    # project cube or model
                    frame = render(frame, obj, projection, ref, False)
                except:
                    pass
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return rvec_list, tvec_list

if __name__ == '__main__':
    multi_marker_pose()
