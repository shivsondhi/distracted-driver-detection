'''
Training code and the image data generators.
'''

# imports
import os
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import cvlib as cv


# Global variables
CLASSES = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]


def single_model_generator(model, filepath, img_dim, b_train, b_val):
	if model == "cnn":
		img_gen = image.ImageDataGenerator()
	elif model == "vgg":
		img_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
	traingen = img_gen.flow_from_directory(os.path.join(filepath, 'train'),
		target_size = (img_dim, img_dim),
		batch_size = b_train,
		class_mode = 'categorical')
	valgen = img_gen.flow_from_directory(os.path.join(filepath, 'val'),
		target_size = (img_dim, img_dim),
		batch_size = b_val,
		class_mode = 'categorical')
	# sys.getsizeof(traingen) = sys.getsizeof(valgen) = 56 bytes
	return traingen, valgen


def ensemble_model_generator(model, dirs, img_dim, batch, t_or_v):
	if model == "cnn":
		img_gen = image.ImageDataGenerator()
	elif model == "vgg":
		img_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
	genX1 = img_gen.flow_from_directory(os.path.join(dirs[0], t_or_v),
										target_size = (img_dim, img_dim),
										batch_size = batch,
										class_mode = 'categorical',
										seed = 7)
	genX2 = img_gen.flow_from_directory(os.path.join(dirs[1], t_or_v),
										target_size = (img_dim-1, img_dim-1),
										batch_size = batch,
										class_mode = 'categorical',
										seed = 7)
	genX3 = img_gen.flow_from_directory(os.path.join(dirs[2], t_or_v),
										target_size = (img_dim-2, img_dim-2),
										batch_size = batch,
										class_mode = 'categorical',
										seed = 7)
	while True:
		X1i = genX1.next()
		X2i = genX2.next()
		X3i = genX3.next()
		yield [X1i[0], X2i[0], X3i[0]], X3i[1]


def single_training(model, mod_name, datapath, savepath, img_dim, b_train, b_val, l_rate, num_epochs, steps, val_steps, loadpath=None):
	if mod_name == "cnn":
		opt = optimizers.Adam(lr = l_rate)
	elif mod_name == "vgg":
		opt = optimizers.RMSprop(lr = l_rate)
	# get data generators
	traingen, valgen = single_model_generator(mod_name, datapath, img_dim, b_train, b_val)

	if loadpath:
		# load model weights
		model.load_weights(loadpath)
	# compile the model
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# create model checkpoint
	checkpoint = ModelCheckpoint(savepath, monitor='val_acc', save_best_only=True)
	callbacks_list = [checkpoint]
	# fit and run model
	model.fit_generator(traingen,
		epochs = num_epochs,
		steps_per_epoch = steps,
		validation_data = valgen,
		validation_steps = val_steps,
		callbacks = callbacks_list,
		verbose = 1)
	return


def ensemble_training(model, mod_name, datapaths, savepath, img_dim, b_train, b_val, l_rate, num_epochs, steps, val_steps, loadpath=None):
	opt = optimizers.Adam(lr = l_rate)
	# get data generators
	traingen = ensemble_model_generator(mod_name, datapaths, img_dim, b_train, "train")
	valgen = ensemble_model_generator(mod_name, datapaths, img_dim, b_val, "val")

	if loadpath:
		# load weight models
		model.load_weights(loadpath)
	# compile the model
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# create model checkpoint
	checkpoint = ModelCheckpoint(savepath, monitor='val_acc', save_best_only=True)
	callbacks_list = [checkpoint]
	# fit and run model
	model.fit_generator(traingen,
		epochs = num_epochs,
		steps_per_epoch = steps,
		validation_data = valgen,
		validation_steps = val_steps,
		callbacks = callbacks_list,
		verbose = 1)
	return


def get_pseudolabels(testpath, loadpath, ens_models, img_dim, num_classes):
	classifications = pd.DataFrame(columns=['img_name', 'c0', 'c1', "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "class"])
	num = 0
	for imgpath in os.listdir(testpath):
		num += 1
		img = cv2.imread(os.path.join(testpath, imgpath))
		img_full = cv2.resize(img, (img_dim, img_dim))
		img_full = np.reshape(img_full, [1, img_dim, img_dim, 3])
		if ens_models[0] == "vgg":
			img_full = preprocess_input(img_full)
		# load grayscale image
		grayimg = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
		# get hand crop
		h = 170
		w = 400
		top = (grayimg.shape[0]//2) - (h//3)
		left = (grayimg.shape[1]//2) - (w//2)
		pixels = grayimg[top:top+h, left:left+w]
		img_hand = cv2.resize(pixels, (img_dim-2, img_dim-2))
		img_hand = np.reshape(img_hand, [1, img_dim-2, img_dim-2, 3])
		if ens_models[0] == "vgg":
			img_hand = preprocess_input(img_hand)
		# get face crop
		max_conf = 0
		ypadding = 100
		xpadding = 75
		cropbottom = (int(img.shape[0]//1.5))
		cropright = (int(img.shape[1]//1.5))
		pixels = img[0:cropbottom, 0:cropright]
		faces, confidences = cv.detect_face(pixels)
		if faces:
			for i, face in enumerate(faces):
				if confidences[i] >= max_conf:
					max_conf = max(confidences[i], max_conf)
					left, top, width, height = face
					if (left - xpadding < 0):
						img_face = pixels[top:top+ypadding+height, 0:width+xpadding]
					else:
						img_face = pixels[top:top+ypadding+height, left-xpadding:left+xpadding+width]
		else:
			img_face = pixels
		img_face = cv2.resize(img, (img_dim-1, img_dim-1))
		img_face = np.reshape(img_face, [1, img_dim-1, img_dim-1, 3])
		if ens_models[0] == "vgg":
			img_face = preprocess_input(img_full)
		# load model weights
		model.load_weights(loadpath)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		result = model.predict([img_full, img_face, img_hand])
		y_class = CLASSES[result.argmax(axis=-1)[0]]
		classifications = classifications.append({'img_name': imgpath,
								'c0': result[0][0],
								'c1': result[0][1],
								"c2": result[0][2],
								"c3": result[0][3],
								"c4": result[0][4],
								"c5": result[0][5],
								"c6": result[0][6],
								"c7": result[0][7],
								"c8": result[0][8],
								"c9": result[0][9],
								"class": y_class}, ignore_index=True)
		print(num, "imagepath: ", imgpath)
	classifications.to_csv("pseudolabels.csv", index=False)
	return
