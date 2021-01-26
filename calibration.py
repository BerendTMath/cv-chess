import os

import cv2
import numpy as np
import glob
import sys
# from Deutschland:  https: // github.com / LearnTechWithUs / Stereo - Vision
def calibrate_camera(path):
    # !/usr/bin/env python

    # *************************************************
    # ***** Parameters for Distortion Calibration *****
    # *************************************************

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpointsR = []  # 2d points in image plane
    imgpointsL = []

    # Start calibration from the camera
    print('Starting calibration for the 2 cameras... ')

    # Extracting path of individual image stored in a given directory
    path1 = os.path.join(path, 'frame1/*.png')
    path2 = os.path.join(path, 'frame2/*.png')

    images1 = glob.glob(path1)
    images2 = glob.glob(path2)

    for pic1,pic2 in zip(images1, images2):
        print(pic1)

        img1 = cv2.imread(pic1)
        img2 = cv2.imread(pic2)

        ChessImaR = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ChessImaL = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                                   (9, 6),
                                                   None)  # Define the number of chees corners we are looking for
        retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                                   (9, 6), None)  # Left side
        if retR and retL:
            objpoints.append(objp)
            cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)

    # Determine the new values for different parameters
    #   Right Side
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                            imgpointsR,
                                                            ChessImaR.shape[::-1], None, None)
    hR, wR = ChessImaR.shape[:2]
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                                (wR, hR), 1, (wR, hR))

    #   Left Side
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                            imgpointsL,
                                                            ChessImaL.shape[::-1], None, None)
    hL, wL = ChessImaL.shape[:2]
    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print('Cameras Ready to use')

    # ********************************************
    # ***** Calibrate the Cameras for Stereo *****
    # ********************************************

    # StereoCalibrate function
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                               imgpointsL,
                                                               imgpointsR,
                                                               mtxL,
                                                               distL,
                                                               mtxR,
                                                               distR,
                                                               ChessImaR.shape[::-1],
                                                               criteria_stereo,
                                                               flags)

    # StereoRectify function
    rectify_scale = 0  # if 0 image croped, if 1 image nor croped
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                      ChessImaR.shape[::-1], R, T,
                                                      rectify_scale,
                                                      (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped
    # initUndistortRectifyMap function
    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  ChessImaR.shape[::-1],
                                                  cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   ChessImaR.shape[::-1], cv2.CV_16SC2)
    return Left_Stereo_Map, Right_Stereo_Map

def capture_calibration_images(vid1, vid2, frame1_folder, frame2_folder):
    count = 0

    while True:
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()

        path_frame1 = frame1_folder
        path_frame2 = frame2_folder

        cv2.imshow('Frame 1', frame1)
        cv2.imshow('Frame 2', frame2)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        retR, cornersR = cv2.findChessboardCorners(gray1, (9, 6),
                                                   None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
        retL, cornersL = cv2.findChessboardCorners(gray2, (9, 6), None)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Done capturing")
            break
        if cv2.waitKey(20) & 0xFF == ord('v'):
            print('V')
            if not retR and retL:
                print('cal. board not detected')
            if not os.path.exists(path_frame1):
                os.makedirs(path_frame1)
            if not os.path.exists(path_frame2):
                os.makedirs(path_frame2)
            status1 = cv2.imwrite(f"{os.path.join(path_frame1, str(count))}.png", frame1)
            status2 = cv2.imwrite(f"{os.path.join(path_frame2, str(count))}.png", frame2)

            print(f'saved 1:{status1}, 2:{status2}')
            cv2.imshow('saved image 1', frame1)
            cv2.imshow('saved image 2', frame2)

            count += 1

def undistort(frame1, frame2, stereo_map1, stereo_map2):
    # undistort, from Deutschland
    Right_nice= cv2.remap(frame1,stereo_map1[0],stereo_map1[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Left_nice= cv2.remap(frame2,stereo_map2[0],stereo_map2[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
    return Left_nice, Right_nice
