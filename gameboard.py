import math
import os

import numpy as np
import requests
import selenium
from selenium import webdriver
import time
from PIL import Image
import io
import hashlib
import glob
import cv2
from matplotlib import pyplot as plt
import pickle
import json
import random
# import torch
# from detectron2.structures import BoxMode
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
from corner_detection.main import joink

import functools

storage_chess_matrix = np.array((
    [[5, 2, 3, 10, 9, 3, 2, 5],
     [1, 1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1],
     [5, 2, 3, 10, 9, 3, 2, 5]]))

storage_frame = None

# could send in 2 frames for better accuracy, if needed
def getgame(frame):
    # ttt = 0
    # chess = 1
    return 1


####################################################
def get_degrees(frame):
    SIZE = 4
    luminance_grid = np.zeros((SIZE, SIZE))

    step_size = int(frame.shape[0] / SIZE)
    offset = 10
    p1y = step_size
    p0y = 0
    pixels_p_tile = step_size ** 2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), cv2.BORDER_DEFAULT)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 4)
    thresh = cv2.Canny(thresh, 1, 230)

    for i in range(SIZE):
        p1x = step_size
        p0x = 0
        for j in range(SIZE):

            tile = thresh[p0x:p1x, p0y:p1y]
            tile = tile[offset:(tile.shape[0] - offset), offset:(tile.shape[0] - offset)]

            avg_tile = round(tile.sum() / pixels_p_tile, 1)
            cv2.putText(tile, f"{avg_tile}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (209, 180, 0, 255),  1)

            ##############################
            if avg_tile > 15:
                # tile = np.ones((tile.shape[0], tile.shape[1]))
                luminance_grid[i,j] = 1

            # cv2.imshow(f'34{i,j} TILE', tile)
            p0x += step_size
            p1x += step_size
        p0y += step_size
        p1y += step_size
    print(luminance_grid)

    rotate = 0
    if luminance_grid[1,0] and luminance_grid[1,3] == 1:
    #     rotate 0 or 180 degrees
        print('0 or 180 degrees')
        row1 = frame[0:step_size,0:frame.shape[0]]
        row2 = frame[frame.shape[0] - step_size:frame.shape[0], 0:frame.shape[0]]
    else:
    #     rotate 90 or 270 degrees
        print('90 or 270 degrees')
        row1 = frame[0:frame.shape[0], 0:step_size]
        row2 = frame[0:frame.shape[0], frame.shape[0] - step_size:frame.shape[0]]
        rotate = -90
    cv2.imshow('row1', row1)
    cv2.imshow('row2', row2)
    # White under
    if row1.sum() > row2.sum():
        rotate += 180
    else:
        pass

    print(f'rotate {rotate}')
    return rotate


degrees_to_rotate = None
def get_move_made(frame):
    global degrees_to_rotate
    if degrees_to_rotate is None:
        degrees_to_rotate = get_degrees(frame)
    (h, w) = frame.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, int(degrees_to_rotate), 1)
    right_frame = cv2.warpAffine(frame, M, (h, w))

    global storage_frame

    if storage_frame is None:
        grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), cv2.BORDER_DEFAULT)
        # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 4)
        # thresh = cv2.Canny(thresh, 1, 230)
        storage_frame = blur
        return None

    #     first compare current frame to old frame, if there's a delta get the move
    grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), cv2.BORDER_DEFAULT)
    # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 4)
    # thresh = cv2.Canny(thresh, 1, 230)
    # cv2.imshow('thresh', thresh)

    difference = cv2.absdiff(storage_frame, blur)
    cv2.imshow('difference', difference)
    ret, difference = cv2.threshold(difference,57,255,cv2.THRESH_BINARY)
    cv2.imshow('difference thres', difference)

    ##########################
    SIZE = 8
    step_size = int(grey.shape[0]/SIZE)
    p1y = step_size
    p0y = 0
    offset = 10
    #######################
    output_frame = np.zeros((496,496))
    pixels_p_tile = step_size ** 2
    #######################
    changed_tiles = list()
    for i in range(SIZE):
        p1x = step_size
        p0x = 0
        for j in range(SIZE):
            # print(f'{p0x},{p1x}, {p0y},{p1y}')v
            tile = difference[p0x:p1x, p0y:p1y]

            tile = tile[offset:(tile.shape[0] - offset), offset:(tile.shape[0] - offset)]
            avg_tile = round(tile.sum() / pixels_p_tile, 1)
            cv2.putText(tile, f"{i, 7 - j}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (209, 180, 0, 255),  1)

            ##############################
            if avg_tile > 8.2:
                tile = np.ones((tile.shape[0], tile.shape[1]))
                changed_tiles.append((i, 7 - j))

            output_frame[p0x + offset:(p1x - offset), p0y + offset:(p1y - offset)] = tile
            #############################
            # cv2.imshow(f'testimg{p0x}{p0y}', tile)
            #############################
            p0x += step_size
            p1x += step_size

        p0y += step_size
        p1y += step_size

    ############## OUTRO ################
    print(changed_tiles)
    cv2.imshow(f'output frame', output_frame)
    move = get_move(changed_tiles)
    # for n in range(SIZE):
    #     for m in range(SIZE):
    #         print(storage_chess_matrix[n,m], end='')
    #         print(" ", end='')
    #     print()
    # print(f'move {move}')

    grey = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), cv2.BORDER_DEFAULT)
    storage_frame = blur

    return move

def get_move(changed_tiles):

    if len(changed_tiles) < 2:
        return None
    zero_tile = changed_tiles[1]
    one_tile = changed_tiles[0]

    # print(f'zero_tile {zero_tile}, one_tile {one_tile}')

    # Convert format
    number_to_letter = ['a','b','c','d','e','f','g','h']
    char0 = number_to_letter[zero_tile[0]]
    char2 = number_to_letter[one_tile[0]]

    # return in format a1b2
    return f'{char0}{zero_tile[1] + 1}{char2}{one_tile[1] + 1}'

def update_storage_chess_matrix(move):
    print(f'move from update function {move}')
    global storage_chess_matrix
    char0 = int([ord(char) - 97 for char in str(move)[0].lower()][0])
    char2 = int([ord(char) - 97 for char in str(move)[2].lower()][0])

    char1 = int(str(move)[3]) - 1
    char3 = int(str(move)[1]) - 1

    print(f'{char2} {char3} - {char0} {char1}')
    storage_chess_matrix[(char2, char3)] = 0
    storage_chess_matrix[(char0, char1)] = 1
    for n in range(8):
        for m in range(8):
            print(storage_chess_matrix[n,m], end='')
            print(" ", end='')
        print()
    print(" ")
def generate_phat(img):
    """crop using 4 points transform"""
    # TODO obtain these from corner detection DONE
    print(img.shape)
    P_in_layer = joink(img)

    for i,P in enumerate(P_in_layer):
        P_in_layer[i] = np.float32(P)
        # print(f'P layer {i}: {P}')


    # P_in_layer0 = np.float32([[408, 88], [407, 469], [-1, 444], [43, 72]])
    # P_in_layer1 = np.float32([[489, 491], [15, 491], [22, 29], [482, 28]])
    # P_in_layer2 = np.float32([[497, 496], [4, 496], [6, 7], [495, 6]])

    # todo write function for this DONE
    # MapP0ToP1 = np.float32([[500, 0], [500, 500], [0, 500], [0, 0]])
    # MapP1ToP2 = np.float32([[500, 500], [0, 500], [0, 0], [500, 0]])

    MapP0ToP1 = generate_map(P_in_layer[0])
    MapP1ToP2 = generate_map(P_in_layer[1])

    H0to1 = cv2.getPerspectiveTransform(P_in_layer[0], MapP0ToP1)
    H1to2 = cv2.getPerspectiveTransform(P_in_layer[1], MapP1ToP2)


    # P_in_layer2_modified = np.float32([[497, 496, 1], [4, 496, 1], [6, 7, 1], [495, 6, 1]]).T

    P_in_layer2_modified = (P_in_layer[2]).T
    # print(P_in_layer2_modified)
    P_in_layer2_modified = np.vstack( (P_in_layer2_modified, np.ones((1,4)) ) )
    # print(P_in_layer2_modified)
    Phat = np.dot(np.linalg.inv(H0to1), np.dot(np.linalg.inv(H1to2), P_in_layer2_modified))

    for col in Phat.T:
        # devide cols by scalar
        col /= col[2]

    Phat = np.delete(Phat, 2, 0)
    Phat = Phat.astype(np.int).T

    return Phat
def generate_map(P_in_layer):
    temp = dict()
    return_list = list()

    for i, item in enumerate(P_in_layer):
        temp[i] = list(item)

    ding = sorted(temp, key=lambda x: temp[x][0], reverse=True)
    for i,joch in enumerate(ding):
        temp[joch][0] = (500 if i < 2 else 0)

    ding = sorted(temp, key=lambda x: temp[x][1], reverse=True)
    for i,joch in enumerate(ding):
        temp[joch][1] = (500 if i < 2 else 0)


    for heer in temp.values():
        return_list.append(heer)
    return_list = np.float32(return_list)
    # print(f'MAP: {return_list}')

    return return_list

def warp_board(img, Phat):
    Phat = Phat.astype(np.float32)

    Map_Phat = np.float32([[500, 500], [0, 500], [0, 0], [500, 0]])

    H_Phat = cv2.getPerspectiveTransform(Phat, Map_Phat)
    warped = cv2.warpPerspective(img, H_Phat, (500, 500))

    return warped

def image_resize(img, height=500):
    pixels = height * height; shape = list(np.shape(img))
    scale = math.sqrt(float(pixels)/float(shape[0]*shape[1]))
    shape[0] *= scale; shape[1] *= scale
    img = cv2.resize(img, (int(shape[1]), int(shape[0])))
    return img


###########################################
def setup_detectron(frame):
    # if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # quit(1)
    try:
        # https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0
        from detectron2.data.datasets import register_coco_instances

        register_coco_instances("microcontroller_train", {}, "./data/chessboard/train.json", "./data/chessboard/train")
        register_coco_instances("microcontroller_test", {}, "./data/chessboard/test.json", "./data/chessboard/test")

        # Visualize/inspect dataset
        # for d in random.sample(dataset_dicts, 2):
        #     img = cv2.imread(d["file_name"])
        #     v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)
        #     v = v.draw_dataset_dict(d)
        #     plt.figure(figsize = (14, 10))
        #     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        #     plt.show()

        # config model

    except:
        print('already done')

    from detectron2.data import DatasetCatalog, MetadataCatalog
    dataset_dicts = DatasetCatalog.get("microcontroller_train")
    microcontroller_metadata = MetadataCatalog.get("microcontroller_train")
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.DATASETS.TRAIN = ("microcontroller_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # train
    print('training')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # print(f'output dir {cfg.OUTPUT_DIR}')
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # inference
    print('inference')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ("microcontroller_test",)

    from detectron2.utils.visualizer import ColorMode
    predictor = DefaultPredictor(cfg)
    # dataset_dicts = DatasetCatalog.get("microcontroller_train")
    # for d in random.sample(dataset_dicts, 3):
    #     print(d)
    # im = cv2.imread(d["file_name"])
    im = frame

    outputs = predictor(im)

    # print(f'output {outputs}')
    # print(f'outputs {outputs["instances"]}')

    # print(f'instances {outputs["instances"].__getitem__(0)}')
    outputs['instances'] = outputs["instances"].__getitem__(0)

    # ding = dict()
    # ding['instances'] = outputs["instances"].__getitem__(0)
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # fig = plt.figure(figsize=(10, 8))
    # plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)))
    cv2.imshow('chess detect', cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))


''' Storage
# SCrape chessboard images
# https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
def fetch_image_urls(query: str, max_links_to_fetch, wd: webdriver, sleep_between_interactions):

    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

        # build the google query

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    print(f'max links {max_links_to_fetch}')
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def scrape():
    DRIVER_PATH = './scraper/chromedriver.exe'
    wd = webdriver.Chrome(executable_path=DRIVER_PATH)


    links = fetch_image_urls('chessboard paper',100, wd, 0.5)
    with open("./scraper/chessboardpaper.txt", "wb") as fp:   #Pickling
        pickle.dump(links, fp)

    # open
    # with open("./scraper/chessgame.txt", "rb") as fp:   # Unpickling
    #     links = pickle.load(fp)

    links = list(links)

    for image in links:
        persist_image('./data/chessboard',image)

# scrape()
def rename():
    path = './data/chessboard/images/'

    print('hit')
    for count, i in enumerate(os.listdir(path)):
        print(i)
        os.rename(os.path.join(path, i), os.path.join(path, f'{count}.jpg'))
# rename()



# '''
# count = 220
# def capture_images(right, path):
#     global count
#
#     cv2.imshow('right 2', right)
#     if cv2.waitKey(20) & 0xFF == ord('v'):
#         print('V')
#         # status1 = cv2.imwrite(f"{os.path.join(path, str(count))}.png", left)
#         # grey = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
#         status2 = cv2.imwrite(f"{os.path.join(path, f'{str(count)}')}.png", right)
#         count += 1
#         print(f'saved 1:{None}, 2:{status2}')
# def resize_imgs():
#     path = './data/chessboard/images/'
#
#     # labels = np.zeros((8,1))
#     print('hit')
#     # training_data = []
#     for i, img in enumerate(os.listdir(path)):
#         global corner_count
#         corner_count = 0
#         print(img)
#
#         img_array = cv2.imread(os.path.join(path, img))
#         new_array = cv2.resize(img_array, (640,480))
#         # fig, ax = plt.subplots()
#         # ax.imshow(new_array, cmap='gray')
#         # fig.canvas.mpl_connect('button_press_event', onclick)
#         # plt.show()
#
#         # training_data.append([new_array, temp_arr])
#         cv2.imwrite(f"{os.path.join('./data/chessboard/resize_color/', f'r{str(i)}')}.png", new_array)
# resize_imgs()

# print(training_data[0][0].shape)
# print(training_data[0][1].shape)

# pickle_out = open('training.pickle', 'wb')
# pickle.dump(training_data, pickle_out)
# pickle_out.close()
#
# print("import")
#
# pickle_in = open('training.pickle', 'rb')
# training_test = pickle.load(pickle_in)
#
# print(training_test[0][0].shape)
# print(training_test[0][1].shape)

# temp_arr = np.zeros((8,1), dtype=np.int)
# def onclick(event):
#
#     global corner_count
#
#     print(corner_count)
#     # print(event.xdata, event.ydata)
#
#     if corner_count == 4:
#         return
#
#     temp_arr[0 + corner_count * 2, 0] = round(event.xdata)
#     temp_arr[1 + corner_count * 2, 0] = round(event.ydata)
#
#     print(temp_arr)
#     corner_count += 1
# def copy_jsons():
#     from_path = './data/chessboard/backup/resize'
#     output_path = './data/chessboard/resize_color'
#     from shutil import copyfile
#
#     for filename in [file for file in os.listdir(from_path) if file.endswith('.json')]:
#         copyfile(os.path.join(from_path, filename), os.path.join(output_path, filename))
# copy_jsons()

# def shuffledata():
#     path = './data/chessboard/resize_color/'
#     print('hit')
#     all_png_files = []
#     for filename in [file for file in os.listdir(path) if file.endswith('.png')]:
#         all_png_files.append(filename)
#     print(len(all_png_files))
#     random.shuffle(all_png_files)
#
#     for i,item in enumerate(all_png_files):
#         try:
#             os.rename(os.path.join(path, item),os.path.join(path, f'{i}.png') )
#             name = item.split('.')[0]
#             os.rename(os.path.join(path, f'{name}.json'), os.path.join(path, f'{i}.json'))
#             # print(item)
#         except:
#             print(item)
# shuffledata()

# def rename_jsons():
#     path = './data/chessboard/resize_color/'
#     for filename in [file for file in os.listdir(path) if file.endswith('.json')]:
#         with open(os.path.join(path, filename), 'r+') as f:
#             data = json.load(f)
#             data['imagePath'] = f'{filename.split(".")[0]}.png'
#             f.seek(0)  # <--- should reset file position to the beginning.
#             json.dump(data, f, indent=4)
#             f.truncate()  # remove remaining part
#
# rename_jsons()


'''

def getboardcorners(frame, game):
    corners = np.zeros((2,4), np.int)
    # ai
    if game == 1:
        # setup_detectron(frame)

        # dst = cv2.Canny(frame, 50, 200, None, 3)
        img = cv2.imread('./data/chessboard/images/141.jpg')
        plt.imshow(img)
        plt.show()
        pass

        # # Copy edges to the images that will display the results in BGR
        # cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        # cdstP = np.copy(cdst)
        #
        # lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        #
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        #
        # linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        #
        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        #
        # cv2.imshow("Source", frame)
        # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    return corners
'''
