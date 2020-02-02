'''
Model code
'''


# import statements
import sys
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.layers.merge import concatenate
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
from matplotlib import pyplot as plt


def set_mode(mode):
	global MODE
	MODE = mode


def simpleCNN(img_dim, num_classes):
	# Create model
	model = Sequential()
	# Add conv layers
	model.add(BatchNormalization(input_shape=(img_dim, img_dim, 3)))
	model.add(Conv2D(64, (5, 5), strides=(3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2, 2)))
	model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2, 2)))
	# Add dense layers
	model.add(Flatten())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Print model summary
	if MODE == "show":
		print("Number of layers in model = ", len(model.layers))
		print("Size of model object (in bytes) = ", sys.getsizeof(model))
		model.summary()
	return model


def vgg16(img_dim, num_classes):
	# img_dim - 1 for face and -2 for hand
	model = VGG16(include_top=False, weights='imagenet', input_shape=(img_dim, img_dim, 3))
	# Freeze last 4 layers of convolutions.
	for layer in model.layers[:-5]:
		layer.trainable = False
	# Common head
	flat1 = Flatten()(model.output)
	classifier = Dense(num_classes, activation='softmax')(flat1)
	model = Model(input = model.input, output = classifier)
	# print model summary
	if MODE == "show":
		print("Number of layers in model = ", len(model.layers))
		print("Size of model object (in bytes) = ", sys.getsizeof(model))
		model.summary()
	return model


def ensemble(ens_models, img_dim, num_classes, loadpaths, plotpaths):
	# Create sub-models
	# pretrained model for face data
	if ens_model[2] == "vgg":
		model1 = vgg16(img_dim-1, num_classes)
	else:
		model1 = simpleCNN(img_dim-1, num_classes)
	# pretrained model for full data
	if ens_model[0] == "vgg":
		model2 = vgg16(img_dim, num_classes)
	else:
		model2 = simpleCNN(img_dim, num_classes)
	# pretrained model for hands data
	if ens_model[1] == "vgg":
		model3 = vgg16(img_dim-2, num_classes)
	else:
		model3 = simpleCNN(img_dim-2, num_classes)

	# Save model plots and print individual summaries
	if MODE == "show":
		model1.summary()
		plot_model(model1, show_shapes=True, to_file=plotpaths[2])
		model2.summary()
		plot_model(model2, show_shapes=True, to_file=plotpaths[0])
		model3.summary()
		plot_model(model3, show_shapes=True, to_file=plotpaths[1])

	# Load saved weights
	model1.load_weights(loadpaths[2])
	model2.load_weights(loadpaths[0])
	model3.load_weights(loadpaths[1])

	# Freeze the models so that they are not trained anymore
	for layer in model1.layers:
		layer.trainable = False
		layer.name = "{}_faces_{}".format(ens_model[2], layer.name)
	for layer in model2.layers:
		layer.trainable = False
		layer.name = "{}_full_{}".format(ens_model[0], layer.name)
	for layer in model3.layers:
		layer.trainable = False
		layer.name = "{}_hands_{}".format(ens_model[1], layer.name)

	# Create final ensemble model
	ensemble_inputs = [model2.input, model1.input, model3.input]
	ensemble_outputs = [model2.output, model1.output, model3.output]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(15, activation='relu')(merge)
	output = Dense(10, activation='softmax')(hidden)
	model = Model(inputs = ensemble_inputs, outputs = output)
	if MODE == "show":
		print("Number of layers in model = ", len(model.layers))
		print("Size of model object (in bytes) = ", sys.getsizeof(model))
		model.summary()
		plot_model(model, show_shapes = True, to_file=plotpaths[3])
	return model
