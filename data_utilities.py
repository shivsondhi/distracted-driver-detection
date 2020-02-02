'''
Utility Functions.
'''


# import statements
import os
import numpy as np
import pandas as pd
import cv2
import random
from mtcnn.mtcnn import MTCNN
import cvlib as cv


# Global variables
CLASSES = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
MODE = "hide"


def set_mode(mode):
	global MODE
	MODE = mode
	return


def create_valset(val_destpath, originpath, move2val):
	if not os.path.exists(val_destpath):
		os.makedirs(val_destpath)
	for d in CLASSES:
		num = 0
		# make class folders
		if not os.path.exists(os.path.join(val_destpath, d)):
			os.makedirs(os.path.join(val_destpath, d))
		# move files to val
		for filename in os.listdir(os.path.join(originpath, d)):
			num += 1
			os.rename(os.path.join(originpath, d, filename), os.path.join(val_destpath, d, filename))
			if (num > (move2val-1)):
				break
		print("End of class {}".format(d))
	return


def create_miniset(mini_destpath, valpath, trainpath, move2mini):
	if not os.path.exists(mini_destpath):
		os.makedirs(mini_destpath)
	for d in CLASSES:
		# make class folders within a train folder
		if not os.path.exists(os.path.join(mini_destpath, "train", d)):
			os.makedirs(os.path.join(mini_destpath, "train", d))
		# make class folders within a val folder
		if not os.path.exists(os.path.join(mini_destpath, "val", d)):
			os.makedirs(os.path.join(mini_destpath, "val", d))
		# move files to miniset
		num = 0
		for filename in os.listdir(os.path.join(trainpath, d)):
			num += 1
			os.rename(os.path.join(trainpath, d, filename), os.path.join(mini_destpath, "train", d, filename))
			if (num > (move2mini[0]-1)):
				break
		num = 0
		for filename in os.listdir(os.path.join(valpath, d)):
			num += 1
			os.rename(os.path.join(valpath, d, filename), os.path.join(mini_destpath, "val", d, filename))
			if (num > (move2mini[1]-1)):
				break
		print("End of class {}".format(d))
	return


def create_dir(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	return


def move_dir():
	original_path = "/content/content/content/faces/"
	new_path = "/content/content/faces/"
	os.rename(original_path, new_path)
	return


def create_dir_struct(basepath):
	subdirs = ["train", "val"]
	create_dir(basepath)
	for subdir in subdirs:
		for c in CLASSES:
			newpath = os.path.join(basepath, subdir, c)
			create_dir(newpath)
		print("Created {} class directories".format(subdir))
	return


def count_files():
	filepath = "/content/split7"
	data_type = "test"

	subdirs = ["train", "val"]
	classes = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

	if data_type == "train":
		for subdir in subdirs:
			length = []
			print("FILENAME: {}".format(os.path.join(filepath, subdir)))
			for c in classes:
				length.append(len([name for name in os.listdir(os.path.join(filepath, subdir, c)) if os.path.isfile(os.path.join(filepath, subdir, c, name))]))
			print("Individual classes: ", length)
			print("Total: ", sum(length), end="\n\n")
	elif data_type == "test":
		print("Number of images: ", len([name for name in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, name))]))
	return


def delete_files():
	croppedpath = "/content/faces"
	classes = ["c9"] # "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",

	for d in classes:
		for filename in os.listdir(os.path.join(croppedpath, "train", d)):
			impath = os.path.join(croppedpath, "train", d, filename)
			os.remove(impath)
		for filename in os.listdir(os.path.join(croppedpath, "val", d)):
			impath = os.path.join(croppedpath, "val", d, filename)
			os.remove(impath)
		print("Deleted from class {}".format(d))
	return


def shift_to(destpath, originpath, num_files, remove=False):
	# number of images to shift per class
	if (num_files[0] == "all") and (num_files[1] == "all"):
		train_ims = val_ims = 10**6
	elif num_files[0] == "all":
		train_ims = 10**6
		val_ims = num_files[1]
	elif num_files[1] == "all":
		train_ims = num_files[0]
		val_ims = 10**6
	else:
		train_ims = num_files[0] 	# 500
		val_ims = num_files[1]		# 100
	for d in CLASSES:
		num = 0
		for filename in os.listdir(os.path.join(originpath, "train", d)):
			if num == train_ims:
				break
			else:
				os.rename(os.path.join(originpath, "train", d, filename), os.path.join(destpath, "train", d, filename))
			num += 1
		num = 0
		if not val_ims:
			print("End of class {}".format(d))
			if remove:
				os.rmdir(os.path.join(originpath, "train", d))
			continue
		for filename in os.listdir(os.path.join(originpath, "val", d)):
			if num == val_ims:
				break
			else:
				os.rename(os.path.join(originpath, "val", d, filename), os.path.join(destpath, "val", d, filename))
			num += 1
		print("End of class {}".format(d))
		if remove:
			os.rmdir(os.path.join(originpath, "train", d))
			os.rmdir(os.path.join(originpath, "val", d))
	if remove:
		os.rmdir(os.path.join(originpath, "train"))
		if not val_ims:
			return
		os.rmdir(os.path.join(originpath, "val"))
	return


def shift_from():
	# variables - number of images to shift per class
	train_ims = 100000
	val_ims = 100000

	# constants
	minipath = "/content/miniset"
	origin_trainpath = "/content/train"
	origin_valpath = "/content/val"
	classes = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

	for d in classes:
		num = 0
		for filename in os.listdir(os.path.join(minipath, "train", d)):
			if num == train_ims:
				break
			else:
				os.rename(os.path.join(minipath, "train", d, filename), os.path.join(origin_trainpath, d, filename))
			num += 1
		num = 0
		for filename in os.listdir(os.path.join(minipath, "val", d)):
			if num == val_ims:
				break
			else:
				os.rename(os.path.join(minipath, "val", d, filename), os.path.join(origin_valpath, d, filename))
			num += 1
		print("End of class {}".format(d))
	return


def compare_images():
	lim = 20
	minipath = "/content/miniset"
	croppedpath = "/content/faces"
	classes = ["c9"] #"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",

	for d in classes:
		num = 0
		for filename in os.listdir(os.path.join(minipath, "train", d)):
			impath = os.path.join(minipath, "train", d, filename)
			display(Image(impath))
			if num == lim:
				break
			num += 1
		num = 0
		for filename in os.listdir(os.path.join(croppedpath, "train", d)):
			impath = os.path.join(croppedpath, "train", d, filename)
			display(Image(impath))
			if num == lim:
				break
			num += 1
	return


def create_hand_crops(originpath, croppedpath):
	val_ims = train_ims = 10**6
	# get crop coords
	impath = os.path.join(originpath, "train", CLASSES[0], random.choice(os.listdir(os.path.join(originpath, "train", CLASSES[0]))))
	img = cv2.imread(impath)
	h = 170
	w = 400
	top = (img.shape[0]//2) - (h//3)
	left = (img.shape[1]//2) - (w//2)

	# crop images, change to grayscale and save in new directory
	for d in CLASSES:
		num = 0
		for filename in os.listdir(os.path.join(originpath, "train", d)):
			impath = os.path.join(originpath, "train", d, filename)
			if num == train_ims:
				break
			else:
				img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
				img = img[top:top+h, left:left+w]
				cv2.imwrite(os.path.join(croppedpath, "train", d, filename), img)
			num += 1
		num = 0
		for filename in os.listdir(os.path.join(originpath, "val", d)):
			impath = os.path.join(originpath, "val", d, filename)
			if num == val_ims:
				break
			else:
				img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
				img = img[top:top+h, left:left+w]
				cv2.imwrite(os.path.join(croppedpath, "val", d, filename), img)
				num += 1
		print("End of class {}".format(d))
	return


def create_face_crops(originpath, croppedpath):
	val_ims = train_ims = 10**6
	max_conf = 0
	ypadding = 100
	xpadding = 75
	# get initial crop coords
	impath = os.path.join(originpath, "train", CLASSES[0], random.choice(os.listdir(os.path.join(originpath, "train", CLASSES[0]))))
	img = cv2.imread(impath)
	cropbottom = (int(img.shape[0]//1.5))
	cropright = (int(img.shape[1]//1.5))
	# crop images, change to grayscale and save in new directory
	for d in CLASSES:
		num = 0
		for filename in os.listdir(os.path.join(originpath, "train", d)):
			impath = os.path.join(originpath, "train", d, filename)
			if num == train_ims:
				break
			else:
				# load image, initial crop and get face coords
				pixels = cv2.imread(impath)
				pixels = pixels[0:cropbottom, 0:cropright]
				faces, confidences = cv.detect_face(pixels)
				if faces:
					for i, face in enumerate(faces):
						if confidences[i] >= max_conf:
							max_conf = max(confidences[i], max_conf)
							left, top, width, height = face
							if (left - xpadding < 0):
								img = pixels[top:top+ypadding+height, 0:width+xpadding]
							else:
								img = pixels[top:top+ypadding+height, left-xpadding:left+xpadding+width]
					cv2.imwrite(os.path.join(croppedpath, "train", d, filename), img)
				else:
					cv2.imwrite(os.path.join(croppedpath, "train", d, filename), pixels)
				max_conf = 0
			num += 1
		num = 0
		for filename in os.listdir(os.path.join(originpath, "val", d)):
			impath = os.path.join(originpath, "val", d, filename)
			if num == val_ims:
				break
			else:
				# load image, initial crop and get face coords
				pixels = cv2.imread(impath)
				pixels = pixels[0:cropbottom, 0:cropright]
				faces, confidences = cv.detect_face(pixels)
				if faces:
					for i, face in enumerate(faces):
						if confidences[i] >= max_conf:
							max_conf = max(confidences[i], max_conf)
							left, top, width, height = face
							if (left - xpadding < 0):
								img = pixels[top:top+ypadding+height, 0:width+xpadding]
							else:
								img = pixels[top:top+ypadding+height, left-xpadding:left+xpadding+width]
					cv2.imwrite(os.path.join(croppedpath, "val", d, filename), img)
				else:
					continue
				max_conf = 0
			num += 1
		print("End of class {}".format(d))
	return
