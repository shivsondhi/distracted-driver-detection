'''
Detect distracted drivers using image data provided by State Farm.
'''

# import statements
import os
import numpy as np
import pandas as pd
import cv2
import random
import tensorflow as tf
from keras import backend as k
import data_utilities as du
from data_utilities import *
import models as mods
from models import *
from generators_and_training import single_training, ensemble_training


def main():
	# PLUGS
	# show or hide extra information?
	modes = ["show", "hide"]
	mode = modes[0]
	# need to do initial data preparation? (Only to be done the first time - True / False)
	data_prep = True
	# do you want to run one of the single models or the ensemble model?
	run_models = ["single", "ensemble"]
	run_model = run_models[0]
	# if single, what dataset do you want to train your single model on?
	datasets = ["main_full_path", "mini_full_path", "main_hands_path", "mini_hands_path", "main_face_path", "mini_face_path"]
	dataset = datasets[1]
	# simpleCNN or VGG16 model?
	train_models = ["simpleCNN", "vgg16"]
	train_model = train_models[1]
	# if ensemble, what dataset do you want to train your ensemble model on?
	datasizes = ["main", "mini"]
	datasize = datasizes[1]
	# fill in the model name [cnn / vgg] that performs the best on each dataset
	ens_models = {"full": "vgg",
		"hands": "vgg",
		"face": "vgg"
	}
	# See README.md for more information about the savepath and loadpaths' naming rules
	# weights file to save the best model's weights to during training
	savepath = "savedModels/[SimpleCNN/vgg16]-[full/hands/faces]_trial[01]-[00]+{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
	# weights file to load when continuing training from a checkpoint
	loadpath = "savedModels/SimpleCNN-full_trial22-20+40-0.29-0.91.hdf5"
	# best weights file trained on full images [used in ensemble model]
	loadpath1 = "savedModels/path/to/file.hdf5"
	# best weights file trained on hand images [used in ensemble model]
	loadpath2 = "savedModels/path/to/file.hdf5"
	# best weights file trained on face images [used in ensemble model]
	loadpath3 = "savedModels/path/to/file.hdf5"
	# File names to save images of the model summaries
	full_model_image = "simpleCNN.png"
	hands_model_image = "vgg16.png"
	face_model_image = "vgg16.png"
	ensemble_image = "ensemble.png"
	# do you want to generate pseudo labels? (True / False)
	pseudolabels = True
	# weights file to use while generating pseudo labels
	pseudo_loadpath = "savedModels/path/to/ensemble_model/weights/file.hdf5"

	# HYPERPARAMETERS
	# train:    full = 16124 | mini = 1800  |   12924
	# val:      full = 3730  | mini = 770   |   3500
	num_epochs = 15
	l_rate = 1e-4
	b_train = 256
	b_val = 128
	train_steps = 51
	val_steps = 28

	# CONSTANTS
	num_classes = 10
	img_dim = 224
	val_num = 30
	move2val_tot = 4500
	move2val_class = val_num // num_classes		# per class
	# [train, val] 180, 77
	move2mini_class = [1, 1] 					# per class
	single_loadpaths = [loadpath1, loadpath2, loadpath3]
	model_plotpaths = [full_model_image, hands_model_image, face_model_image, ensemble_image]

	# Set global MODE in utility files
	mods.set_mode(mode)
	du.set_mode(mode)

	# FILE PATHS
	filepaths = {}
	basepath = "C:\\path\\to\\unzipped\\data\\state-farm-distracted-driver-detection\\imgs"
	testpath = os.path.join(basepath, "test")		# path to test images
	filepaths["main_full_path"] = os.path.join(basepath, "full_split", "mainset")
	filepaths["train_full_path"] = os.path.join(filepaths["main_full_path"], "train")
	filepaths["val_full_path"] = os.path.join(filepaths["main_full_path"], "val")
	filepaths["mini_full_path"] = os.path.join(basepath, "full_split",  "miniset")
	filepaths["main_hands_path"] = os.path.join(basepath, "hand_split", "mainset")
	filepaths["mini_hands_path"] = os.path.join(basepath, "hand_split", "miniset")
	filepaths["main_face_path"] = os.path.join(basepath, "face_split", "mainset")
	filepaths["mini_face_path"] = os.path.join(basepath, "face_split", "miniset")

	if data_prep:
		# create full data directory structures and remove origin train path
		create_dir_struct(filepaths["main_full_path"])
		shift_to(filepaths["main_full_path"], basepath, ["all", 0], remove=True)
		# delete old train directory
		# create cropped data directory structures
		create_dir_struct(filepaths["main_hands_path"])
		create_dir_struct(filepaths["mini_hands_path"])
		create_dir_struct(filepaths["main_face_path"])
		create_dir_struct(filepaths["mini_face_path"])
		# DATA PREPARATION
		# Split training data into train and validation
		create_valset(filepaths["val_full_path"], filepaths["train_full_path"], move2val_class)
		create_miniset(filepaths["mini_full_path"], filepaths["val_full_path"], filepaths["train_full_path"], move2mini_class)
		# create data from main train and val
		create_hand_crops(filepaths["main_full_path"], filepaths["main_hands_path"])
		create_face_crops(filepaths["main_full_path"], filepaths["main_face_path"])
		# create data from mini train and val
		create_hand_crops(filepaths["mini_full_path"], filepaths["mini_hands_path"])
		create_face_crops(filepaths["mini_full_path"], filepaths["mini_face_path"])
		print("~End of Data Preparation phase~")

	if run_model == "single":
		# CREATE & TRAIN MODEL ON THE SELECTED DATASET
		if train_model == "simpleCNN":
			sc_model = simpleCNN(img_dim, num_classes)
			# Simple CNN on selected dataset
			single_training(model = sc_model,
				mod_name = "cnn",
				datapath = filepaths[dataset],
				savepath = savepath,
				img_dim = img_dim,
				b_train = b_train,
				b_val = b_val,
				l_rate = l_rate,
				num_epochs = num_epochs,
				steps = train_steps,
				val_steps = val_steps
			)
			print("SimpleCNN trained on {}.\n\n".format(filepaths[dataset]))
		elif train_model == "vgg16":
			vgg_model = vgg16(img_dim, num_classes)
			single_training(model = vgg_model,
				mod_name = "vgg",
				datapath = filepaths[dataset],
				savepath = savepath,
				img_dim = img_dim,
				b_train = b_train,
				b_val = b_val,
				l_rate = l_rate,
				num_epochs = num_epochs,
				steps = train_steps,
				val_steps = val_steps
			)
			print("VGG16 trained on {}.\n\n".format(filepaths[dataset]))
	elif run_model == "ensemble":
		# CREATE ENSEMBLE MODEL
		ensemble_model = ensemble(ens_models, img_dim, num_classes, single_loadpaths, model_plotpaths)
		# TRAIN ENSEMBLE MODEL
		if datasize == "main":
			ensemble_training(model = ensemble_model,
				datapaths = [main_full_path, main_face_path, main_hands_path],
				savepath = savepath,
				img_dim = img_dim,
				b_train = b_train,
				b_val = b_val,
				l_rate = l_rate,
				num_epochs = num_epochs,
				steps = train_steps,
				val_steps = val_steps
			)
		elif datasize == "mini":
			ensemble_training(model = ensemble_model,
				mod_names = [],
				datapaths = [mini_full_path, mini_face_path, mini_hands_path],
				savepath = savepath,
				img_dim = img_dim,
				b_train = b_train,
				b_val = b_val,
				l_rate = l_rate,
				num_epochs = num_epochs,
				steps = train_steps,
				val_steps = val_steps
			)
	# GENERATE PSEUDO LABELS
	if pseudolabel:
		get_pseudolabels(testpath = testpath,
			loadpath = pseudo_loadpath,
			ens_models = ens_models,
			img_dim = img_dim,
			num_classes = num_classes
		)


if __name__ == "__main__":
	main()
