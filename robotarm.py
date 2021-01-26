import serial
import numpy as np
import cv2
import math
zero_point, chess_grid, pixel_per_mm = None, None, None

def calibrate_arm(frame):
    # detect red square or circle (calibration object)
    object_length = 2.5 #in cm
    #
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60, 55, 55])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('mask', mask)

    dst = cv2.cornerHarris(mask, 17, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(mask, np.float32(centroids), (5, 5), (-1, -1), criteria)
    for i in range(1, len(corners)):
        # print(corners[i])
        cv2.circle(frame, (int(corners[i, 0]), int(corners[i, 1])), 7, (0, 255, 0), 2)

    # frame[dst > 0.1 * dst.max()] = [0, 0, 255]
    cv2.imshow('image', frame)

    # print(corners[1]-corners[2])
    pixel_distance0 = np.square(corners[1]-corners[2]).sum() ** 0.5 + 2.1
    pixel_distance1 = np.square(corners[3]-corners[4]).sum() ** 0.5 + 2.1

    # print(f'2 {corners[3]}, 3 {corners[4]}')
    # print(corners[3] - corners[4])
    # pixel_distance2 = np.square(corners[1]-corners[3]).sum() ** 0.5 + 2.1
    # pixel_distance3 = np.square(corners[2]-corners[4]).sum() ** 0.5 + 2.1

    # print(pixel_distance0)
    # print(pixel_distance1)
    # print(pixel_distance2)
    # print(pixel_distance3)

    avg_corners = (pixel_distance0 + pixel_distance1) / 2
    print(f'avg: {avg_corners}')

    # avf_Corners = 40 mm
    # 40mm/avg_corners = mm per pixel

    pixel_per_mm = 40 / avg_corners

    # scale frame by mm_per_pixel ( (500x500) * mm_per_pixel )
    # scale frame
    scaledFrame = cv2.resize(frame, (int(500 * pixel_per_mm),int(500 * pixel_per_mm) ))
    print(f'scaledd frame {scaledFrame.shape}')
    # cv2.imshow('scaledFrame', scaledFrame)

#     find zero point
    center_square = (int(corners[1, 0] + 0.5 * avg_corners),int(corners[1, 1] + 0.5 * avg_corners) )
    print(f'center_square {center_square}')

    np.save('./data/pixel_per_mm', pixel_per_mm)
    np.save('./data/center_square', center_square)


def convert_chess_to_pos(move):
    try:
        chess_grid = np.load('./data/chess_grid.npy')
    except:
        chess_grid = None

    if chess_grid is None:
        chess_grid = np.zeros((8,8,2))
        ypos = (500/8) /2
        stepsize = 500/8
        for i in range(8):
            xpos = (500 / 8) / 2
            for j in range(8):
                chess_grid[i,j] = xpos, ypos
                xpos += stepsize
            ypos += stepsize
        np.save('./data/chess_grid', chess_grid)
    # print(chess_grid[:,:,1])
    # print()
    # print(chess_grid[:,:,0])

    # print(move)
    char0 = int([ord(char) - 97 for char in str(move)[0].lower()][0])
    char2 = int([ord(char) - 97 for char in str(move)[2].lower()][0])

    char1 = 8- int(str(move)[1])
    char3 = 8- int(str(move)[3])

    print(f'x1,y1,x2,y2 {char1},{char0} - {char3},{char2}')
    x1,y1 = chess_grid[char1, char0, 0], chess_grid[char1, char0, 1]
    x2,y2 = chess_grid[char3, char2,0], chess_grid[char3, char2, 1]
    return x1,y1,x2,y2

def move_arm(x1,y1, x2,y2):
    # Input x,y in 500,500 space
    try:
        pixel_per_mm = np.load('./data/pixel_per_mm.npy')
        zero_point = np.load('./data/zero_point.npy')
    except:
        zero_point = None
        pixel_per_mm = None

    if zero_point is None and pixel_per_mm is None:
        return None
    else:
        print(f'moving piece from {x1, y1} to {x2, y2}')

    #     Let center_square = (0, 10, 0) in robot arm coordinates
    # So if zero_point = (183, 296), in robot coordinates this will be 183 - 183 and 296 - (296 + 10)
    y_offset = 10

    X1 = (x1 * pixel_per_mm) - zero_point[0]
    Y1 = (y1 * pixel_per_mm) - (zero_point[1] + y_offset)
    X2 = (x2 * pixel_per_mm) - zero_point[0]
    Y2 = (y2 * pixel_per_mm) - (zero_point[1] + y_offset)

    print(f'moving piece realy from {X1, Y1} to {X2, Y2}')
    # ser = serial.Serial('COM5', 9800, timeout=1)
    ser = serial.Serial(0)  # open first serial port
    print(ser.portstr )  # check which port was really used
    ser.write(f"{X1, Y1, X2,Y2}")  # write a string
    ser.close()
