import os

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
import sys

from calibration import calibrate_camera, capture_calibration_images, undistort
from stereo_vision import depthmap
from gameboard import getgame, generate_phat, warp_board, image_resize, get_move_made, update_storage_chess_matrix
from chessAI import get_best_move, is_move_legal, print_board, update_board, get_move_direction
from robotarm import move_arm, calibrate_arm, convert_chess_to_pos
from tts import speak

def capture_event(event,x,y,flags,params):
    # event= click of the mouse
    # x,y are position of cursor
    # check the event if it was right click
    try:
        center_square = np.load('./data/center_square.npy')
        pixel_per_mm =np.load('./data/pixel_per_mm.npy')
    except:
        return None

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(int((x - center_square[0]) * pixel_per_mm), int((y  - center_square[1] + 10) * pixel_per_mm))
        print(f'real {x, y}')

if __name__ == "__main__":
    # admin & config
    CamR = cv2.VideoCapture(1)  # Wenn 0 then Right Cam and wenn 2 Left Cam
    CamL = cv2.VideoCapture(0)
    if len(sys.argv) > 1:
        # calibrate camera, user can capture multiple images of the cal. chessgrid using the 'v' key
        if sys.argv[1] == "c":
            print("Capturing calibration data")
            capture_calibration_images(CamL, CamR, './data/cali-images/frame1', './data/cali-images/frame2')
            # obtain mtx & dist
            stereo_map1, stereo_map2 = calibrate_camera('./data/cali-images/')
            print(f'type: {type(stereo_map1)}, 2: {type(stereo_map2)}')
            np.save('./data/camera-params/stereo_map1.0', stereo_map1[0])
            np.save('./data/camera-params/stereo_map1.1', stereo_map1[1])

            np.save('./data/camera-params/stereo_map2.0', stereo_map2[0])
            np.save('./data/camera-params/stereo_map2.1', stereo_map2[1])

        if sys.argv[1] == "m":
            os.remove('./data/phat1.npy')

    # load in camera params
    Left_Stereo_Map = (
    np.load('./data/camera-params/stereo_map1.0.npy'), np.load('./data/camera-params/stereo_map1.1.npy'))
    Right_Stereo_Map = (
    np.load('./data/camera-params/stereo_map2.0.npy'), np.load('./data/camera-params/stereo_map2.1.npy'))

    Phat = None

    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        # Rectify the images on rotation and alignement
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                              0)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT, 0)

        # Disparity view #
        # dmap = depthmap(Right_nice, Left_nice)
        # cv2.imshow('Filtered Color Depth', dmap)

        # rectified view
        # cv2.imshow('Nice L', Left_nice)
        cv2.imshow('nice r', Right_nice)

        # Normal view
        # cv2.imshow('frame L', frameL)
        cv2.imshow('frane r', frameR)
        # ##########################

        # Corner Detection
        if Phat is None:
            try:
                Phat = np.load('./data/phat1.npy')
            except:
                Phat = generate_phat(cv2.cvtColor(Right_nice, cv2.COLOR_BGR2RGB))
                print('PHAT')
                np.save('./data/phat1', Phat)
                print(f'coordinates {Phat}')

        warp_right = image_resize(Right_nice)
        warp_right = warp_board(warp_right, Phat)
        cv2.imshow('warp r', warp_right)
        cv2.setMouseCallback('warp r', capture_event)

        try:
            np.load('./data/center_square.npy')
            np.load('./data/chess_grid.npy')
            np.load('./data/pixel_per_mm.npy')
        except:
            calibrate_arm(warp_right)
        # in future, replace v with timer hit
        if cv2.waitKey(20) & 0xFF == ord('v'):
            move_made = get_move_made(warp_right)
        #     if move is valid:
            if move_made is not None:
                move_made = get_move_direction(move_made)
                if is_move_legal(move_made):
                    
                    update_board(move_made)
                    print_board()

                    aimove = get_best_move()
                    speak(aimove)
                    print(f'AI move {aimove}')

                    x1, y1, x2, y2 = convert_chess_to_pos(aimove)
                    move_arm(x1,y1, x2,y2)
                    print_board()
                else:
                    print("UNVALID MOVE")

        ##########################
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Release the Cameras
    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()



# STORAGE:
# shows edges on unwarped frame
# edges_right = image_resize(Right_nice)
# edges_right = cv2.circle(edges_right, tuple(Phat[0]), 20, (255,0,0), 3)
# edges_right = cv2.circle(edges_right, tuple(Phat[1]), 20, (255,0,0), 3)
# edges_right = cv2.circle(edges_right, tuple(Phat[2]), 20, (255,0,0), 3)
# edges_right = cv2.circle(edges_right, tuple(Phat[3]), 20, (255,0,0), 3)
#
# edges_right = image_resize(Right_nice)
# cv2.imshow('edges r', edges_right)
# break
'''
    count = 1
    avr_dmap = np.zeros((480, 640, 3), dtype=np.float)
    pile_dmap = np.zeros((480, 640, 3), dtype=np.float)
    ones = np.ones((480, 640, 3), dtype=np.float)
    
    # numpy goes from 0 - 1 => opencv 0 - 255
    avr_dmap += (dmap / 255) / count,l
    pile_dmap += dmap
    avr_dmap = pile_dmap / count
    avr_dmap /= 255
    count+=1
    cv2.imshow('Average dmap', avr_dmap)

'''


