import os
import sys
import time

import cv2

import numpy as np
import pandas as pd
from data_util import load_data, images_from_data_dict, df_to_array, evenly_distribute_classes
import pickle
from network import ConvolutionalNeuralNetwork
from tensorflow.keras.utils import to_categorical
import tensorflow.keras

if __name__ == "__main__":
	# types_of_models = ["LeNet5", "ResNet", "VGG16"]
	types_of_models = ["VGG16"]
	# types_of_models = ["VGG16"]
	reshape_size = (32, 64, 3)
	data = pd.read_hdf("train_data.hd5", key="train")
	train_x, train_y = df_to_array(data, img_shape=reshape_size)
	temp_train_x, temp_train_y = evenly_distribute_classes(train_x, train_y)
	temp_train_y = to_categorical(temp_train_y)
	
	data = pd.read_hdf("test_data.hd5", key="test")
	test_x, test_y = df_to_array(data, img_shape=reshape_size)
	temp_test_x, temp_test_y = evenly_distribute_classes(test_x, test_y)
	temp_test_y = to_categorical(temp_test_y)
	for i in types_of_models:
		preds = [0, 0]
		best_test_acc = 0
		model_hist = None
		model_type = i
		num_samples = 20000
		learn_rate = 0.001
		if i == "VGG16":
			learn_rate = 0.00005
		model = ConvolutionalNeuralNetwork(input_shape=reshape_size, model_type=model_type,
		                                   output_size=11, learning_rate=learn_rate)
		count = 0
		hist = []
		testing_results = []
		for _ in range(50):
			print(f"Training Session: {count}")
			t_num = int((num_samples + (num_samples * (count/100))))
			if t_num > temp_train_x.shape[0]:
				t_num = temp_train_x.shape[0]
			idx = np.random.choice(temp_train_x.shape[0], t_num, replace=True)
			temp_train_x, temp_train_y = evenly_distribute_classes(train_x, train_y)
			temp_train_y = to_categorical(temp_train_y)
			# test_idx = np.random.choice(test_x.shape[0], 2000, replace=False)
			# if model_type == "VGG16":
			# 	# temp_train_x = tensorflow.keras.applications.vgg16.preprocess_input(temp_train_x)
			# 	# temp_train_y = tensorflow.keras.applications.vgg16.preprocess_input(temp_train_y)
			# 	# model.model.fit(x=temp_x, y=temp_y, epochs=2, batch_size=32, validation_split=0.1)
			# 	model.model.fit(x=temp_train_x[idx], y=temp_train_y[idx], epochs=2, batch_size=32, validation_split=0.05)
			# else:
			model.model.fit(x=temp_train_x[idx], y=temp_train_y[idx], epochs=2, batch_size=32, validation_split=0.05)
			hist.append(model.model.history.history)
			model_hist = model.model.history.history['accuracy'][-1]
			temp_test_x, temp_test_y = evenly_distribute_classes(test_x, test_y)
			temp_test_y = to_categorical(temp_test_y)
			preds = model.model.evaluate(x=temp_test_x, y=temp_test_y)
			testing_results.append(preds)
			if preds[1] > 0.9:
				if preds[1] > np.round(best_test_acc, 3):
					best_test_acc = np.round(preds[1], 3)
					model.model.save(f"Trained_Model_{model_type}_Acc_{preds[1]:.3f}_Sess_{count}.h5", save_format="h5")
					with open(f"Trained_Model_{model_type}_Acc_{preds[1]:.3f}_Sess_{count}.pkl", "wb") as output_file:
						pickle.dump(hist, output_file)
						output_file.close()
					with open(f"Trained_Model_{model_type}_Acc_{preds[1]:.3f}_Sess_{count}_Testing_Results.pkl", "wb") as output_file:
						pickle.dump(testing_results, output_file)
						output_file.close()
			count += 1
		print(f"Sess ended at: {count}")
		model.model.save(f"Trained_Model_{model_type}_Acc_{preds[1]:.3f}_Sess_{count}.h5", save_format="h5")
		with open(f"Trained_Model_{model_type}_Final_Sess_{count}.pkl", "wb") as output_file:
			pickle.dump(hist, output_file)
			output_file.close()
		with open(f"Trained_Model_{model_type}_Acc_{preds[1]:.3f}_Final_Sess_{count}_Testing_Results.pkl",
		          "wb") as output_file:
			pickle.dump(testing_results, output_file)
			output_file.close()


	# which = "train"
	# data = load_data(which)
	# location = ""
	# if which == "test":
	# 	location = "testing images"
	# elif which == "train":
	# 	location = "training images"
	# elif which == "extra":
	# 	location = "extra images"
	# data_as_array = images_from_data_dict(data_dict=data, which_images="sub", reshape_size=reshape_size,
	#                                       number_to_load=[0,-1], load_negatives=True, loc=location, blur_images=False)
	# data_as_array_blurred = images_from_data_dict(data_dict=data, which_images="sub", reshape_size=reshape_size,
	#                                               number_to_load=[0,-1], load_negatives=False, loc=location, blur_images=True)
	#
	# combined_data = np.vstack((data_as_array, data_as_array_blurred))
	# print()
	#
	# df = pd.DataFrame(combined_data)
	#
	# if which == "test":
	# 	df.to_hdf(f"{which}_data.hd5", key="test", mode="w")
	# elif which == "train":
	# 	df.to_hdf(f"{which}_data.hd5", key="train", mode="w")
	# elif which == "extra":
	# 	df.to_hdf(f"{which}_data.hd5", key="extra", mode="w")
	# print("Finished")
