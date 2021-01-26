import gc, os, sys, glob, argparse, corner_detection.utils as utils
print("<<< \x1b[5;32;40m neural-chessboard \x1b[0m >>>")

from corner_detection.config import *
from corner_detection.utils import ImageObject
from corner_detection.slid import pSLID, SLID, slid_tendency #== step 1
from corner_detection.laps import LAPS                       #== step 2
from corner_detection.llr import LLR, llr_pad                #== step 3

from keras import backend as K
import cv2; load = cv2.imread
save = cv2.imwrite
import numpy as np
#NC_SCORE = -1

################################################################################

def layer():
	global NC_LAYER, NC_IMAGE#, NC_SCORE
	global pointsss

	print(utils.ribb("==", sep="="))
	print(utils.ribb("[%d] LAYER " % NC_LAYER, sep="="))
	print(utils.ribb("==", sep="="), "\n")

	# --- 1 step --- find all possible lines (that makes sense) ----------------
	print(utils.ribb(utils.head("SLID"), utils.clock(), "--- 1 step "))
	segments = pSLID(NC_IMAGE['main'])
	raw_lines = SLID(NC_IMAGE['main'], segments)
	lines = slid_tendency(raw_lines)

	# --- 2 step --- find interesting intersections (potentially a mesh grid) --
	print(utils.ribb(utils.head("LAPS"), utils.clock(), "--- 2 step "))
	points = LAPS(NC_IMAGE['main'], lines)
	#print(abs(49 - len(points)), NC_SCORE)
	#if NC_SCORE != -1 and abs(49 - len(points)) > NC_SCORE * 4: return
	#NC_SCORE = abs(49 - len(points))

	# --- 3 step --- last layer reproduction (for chessboard corners) ----------
	print(utils.ribb(utils.head(" LLR"), utils.clock(), "--- 3 step "))
	inner_points = LLR(NC_IMAGE['main'], points, lines)
	four_points = llr_pad(inner_points, NC_IMAGE['main']) # padcrop

	# --- 4 step --- preparation for next layer (deep analysis) ----------------
	print(utils.ribb(utils.head("   *"), utils.clock(), "--- 4 step "))
	print(four_points)



	img_crop = cv2.circle(NC_IMAGE['main'], tuple(four_points[0]), 20, (255, 0, 0), 3)
	img_crop = cv2.circle(NC_IMAGE['main'], tuple(four_points[1]), 20, (255, 0, 0), 3)
	img_crop = cv2.circle(NC_IMAGE['main'], tuple(four_points[2]), 20, (255, 0, 0), 3)
	img_crop = cv2.circle(NC_IMAGE['main'], tuple(four_points[3]), 20, (255, 0, 0), 3)
	from matplotlib import pyplot as plt
	plt.imshow(img_crop)
	plt.show()
	# plt.imshow(NC_IMAGE['orig'])
	# plt.show()
	try:
		NC_IMAGE.crop(four_points)
		print("FOUR POINTS SUCCES")
		return four_points
	except:
		utils.warn("niestety, ale kolejna warstwa nie jest potrzebna")
		NC_IMAGE.crop(inner_points)
		print("INNER POINTS SUCCES")

		return inner_points

################################################################################


def detect(args):
	global NC_LAYER, NC_IMAGE, NC_CONFIG

	# if (not os.path.isfile(args)):
	# 	utils.errn("error: the file \"%s\" does not exits" % args)

	# NC_IMAGE, NC_LAYER = ImageObject(load(args)), 0
	P_in_layer = list()
	NC_IMAGE, NC_LAYER = ImageObject(args), 0
	for _ in range(NC_CONFIG['layers']):
		NC_LAYER += 1
		points = layer()
		# print(points)
		P_in_layer.append(points)

	# save('./data/asdfasdf.jpg', NC_IMAGE['orig'])
	return P_in_layer
	# print("DETECT: %s" % args.input)

def dataset(args):
	print("DATASET: use dataset.py") # FIXME

def train(args):
	print("TRAIN: use train.py") # FIXME

def test(args):
	files = glob.glob('test/in/*.jpg')

	for iname in files:
		oname = iname.replace('in', 'out')
		args.input = iname; args.output = oname
		detect(args)

	print("TEST: %d images" % len(files))
	
################################################################################

if __name__ == "__main__":
	utils.reset()

	p = argparse.ArgumentParser(description=\
	'Find, crop and create FEN from image.')

	p.add_argument('mode', nargs=1, type=str, \
			help='detect | dataset | train')
	p.add_argument('--input', type=str, \
			help='input image (default: input.jpg)')
	p.add_argument('--output', type=str, \
			help='output path (default: output.jpg)')

	#os.system("rm test/steps/*.jpg") # FIXME: to jest bardzo grozne
	os.system("rm -rf test/steps; mkdir test/steps")

	args = p.parse_args(); mode = str(args.mode[0])
	modes = {'detect': detect, 'dataset': dataset, 'train': train, 'test': test}

	if mode not in modes.keys():
		utils.errn("hey, nie mamy takiej procedury!!! (wybrano: %s)" % mode)

	modes[mode](args); print(utils.clock(), "done")
	K.clear_session(); gc.collect() # FIX: tensorflow#3388

def joink(frame):
	utils.reset()
	points = detect(frame)

	# print(f'points from joink {points}')
	return points