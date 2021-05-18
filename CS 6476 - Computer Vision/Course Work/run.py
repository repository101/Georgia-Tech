import os
import sys
import time

import numpy as np
import pandas as pd
import cv2
import tensorflow
from data_util import load_data, images_from_data_dict, df_to_array, evenly_distribute_classes
import pickle
from tensorflow.keras.utils import to_categorical


def load_vgg():
	return tensorflow.keras.models.load_model('Trained_Model_VGG16_Acc_0.957_Sess_49.h5')


def load_lenet():
	return tensorflow.keras.models.load_model('Trained_Model_LeNet5_Acc_0.922_Sess_44.h5')


def calc_IOU(img, bbox_1, bbox_2):
	box_1 = ((bbox_1[0], bbox_1[1]), (bbox_1[2], bbox_1[3]), 0)
	box_2 = ((bbox_2[0], bbox_2[1]), (bbox_2[2], bbox_2[3]), 0)
	retVal, region = cv2.rotatedRectangleIntersection(box_1, box_2)
	if retVal == cv2.INTERSECT_NONE:
		return 0
	elif retVal == cv2.INTERSECT_FULL:
		return 1
	else:
		intersection = cv2.contourArea(region)
		area_1 = bbox_1[2] * bbox_1[3]
		area_2 = bbox_2[2] * bbox_2[3]
		return intersection / (area_1 + area_2 - intersection)
	
	
def process_image_one():
	# Different Scale
	reshape_size = (32, 64, 3)
	input_file_name = 'img1.png'
	image_1_save_file_name = "1.png"
	img = cv2.imread(input_file_name)
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('delta', 'image', 5, 255, nothing)
	# cv2.createTrackbar('min_area', 'image', 97, 10000, nothing)
	# cv2.createTrackbar('max_area', 'image', 150, 6000, nothing)
	# cv2.createTrackbar('max_variation', 'image', 14, 100, nothing)
	# cv2.createTrackbar('edge_blur_size', 'image', 5, 30, nothing)
	#
	# while (1):
	# 	temp_image = np.copy(img)
	# 	gray = cv2.cvtColor(np.copy(temp_image), cv2.COLOR_BGR2GRAY)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	delta = cv2.getTrackbarPos('delta', 'image')
	# 	min_area = cv2.getTrackbarPos('min_area', 'image')
	# 	max_area = cv2.getTrackbarPos('max_area', 'image')
	# 	max_variation = cv2.getTrackbarPos('max_variation', 'image')
	# 	edge_blur_size = cv2.getTrackbarPos('edge_blur_size', 'image')
	# 	max_variation /= 100.0
	#
	# 	vis = np.copy(temp_image)
	# 	if edge_blur_size % 2 == 1 and edge_blur_size > 0:
	# 		mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
	# 		                       _edge_blur_size=edge_blur_size)
	# 		regions, _ = mser.detectRegions(gray)
	# 		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	# 		cv2.polylines(vis, hulls, 1, (0, 255, 0))
	# 	cv2.imshow('image', vis)
	# cv2.destroyAllWindows()
	
	vgg_model = load_vgg()
	lenet_model = load_lenet()
	mser = cv2.MSER_create(_delta=5, _min_area=97, _max_area=150, _max_variation=0.14, _edge_blur_size=5)
	# mser = cv2.MSER_create()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	regions, bounding_boxes = mser.detectRegions(gray)
	
	avail_idx = {t for t in range(bounding_boxes.shape[0])}
	keep = set()
	for i in range(len(bounding_boxes)):
		temp = tuple(bounding_boxes[i])
		if temp not in keep:
			keep.add(temp)
	
	cropped_images = {}
	sub_image_count = 0
	for region in keep:
		region = np.asarray(list(region))
		pad = 5
		region[0] -= pad
		region[1] -= pad
		region[2] += pad
		region[3] += pad
		cropped = np.copy(img)[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
		resized_cropped = cv2.resize(np.copy(cropped), (64, 32), None, 0, 0, cv2.INTER_CUBIC)
		resized_cropped = cv2.GaussianBlur(resized_cropped, ksize=(3, 3), sigmaX=1, sigmaY=1,
		                                   borderType=cv2.BORDER_REFLECT_101)
		complete_prediction_vgg = vgg_model.predict(np.asarray([resized_cropped]))
		complete_prediction_lenet = lenet_model.predict(np.asarray([resized_cropped]))
		vgg_pred = np.argmax(complete_prediction_vgg, axis=1)
		lenet_pred = np.argmax(complete_prediction_lenet, axis=1)
		
		cropped_images[f"img_{sub_image_count}"] = {"region": region, "cropped image": cropped,
		                                            "Complete LeNet Prediction": complete_prediction_lenet,
		                                            "LeNet Prediction": lenet_pred,
		                                            "Complete VGG Prediction": complete_prediction_vgg,
		                                            "VGG Prediction": vgg_pred}
		sub_image_count += 1
	
	resulting_image = np.copy(img)
	for key, val in cropped_images.items():
		pt_1 = (val["region"][0], val["region"][1])
		pt_2 = val["region"][:2] + val["region"][2:]
		cv2.rectangle(resulting_image, pt1=tuple(pt_1), pt2=tuple(pt_2), color=(0, 225, 0), thickness=1)
		cv2.putText(resulting_image, text=f"{val['VGG Prediction'][0]}",
		            org=((pt_2[0]) - 10, int((pt_1[1] + pt_2[1]) / 2) - 25),
		            fontScale=0.75, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 245, 0), thickness=1)
	cv2.imwrite(image_1_save_file_name, resulting_image)
	return


def process_image_two():
	# Different Orientation
	reshape_size = (32, 64, 3)
	input_file_name = 'img2.png'
	image_2_save_file_name = "2.png"
	img = cv2.imread(input_file_name)
	
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('delta', 'image', 24, 255, nothing)
	# cv2.createTrackbar('min_area', 'image', 89, 10000, nothing)
	# cv2.createTrackbar('max_area', 'image', 301, 6000, nothing)
	# cv2.createTrackbar('max_variation', 'image', 28, 100, nothing)
	# cv2.createTrackbar('edge_blur_size', 'image', 5, 30, nothing)
	#
	# while (1):
	# 	temp_image = np.copy(img)
	# 	gray = cv2.cvtColor(np.copy(temp_image), cv2.COLOR_BGR2GRAY)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	delta = cv2.getTrackbarPos('delta', 'image')
	# 	min_area = cv2.getTrackbarPos('min_area', 'image')
	# 	max_area = cv2.getTrackbarPos('max_area', 'image')
	# 	max_variation = cv2.getTrackbarPos('max_variation', 'image')
	# 	edge_blur_size = cv2.getTrackbarPos('edge_blur_size', 'image')
	# 	max_variation /= 100.0
	#
	# 	vis = np.copy(temp_image)
	# 	if edge_blur_size % 2 == 1 and edge_blur_size > 0:
	# 		mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
	# 		                       _edge_blur_size=edge_blur_size)
	# 		regions, _ = mser.detectRegions(gray)
	# 		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	# 		cv2.polylines(vis, hulls, 1, (0, 255, 0))
	# 	cv2.imshow('image', vis)
	# cv2.destroyAllWindows()
	
	vgg_model = load_vgg()
	lenet_model = load_lenet()
	mser = cv2.MSER_create(_delta=24, _min_area=89, _max_area=301, _max_variation=0.28, _edge_blur_size=5)
	# mser = cv2.MSER_create()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	regions, bounding_boxes = mser.detectRegions(gray)
	
	avail_idx = {t for t in range(bounding_boxes.shape[0])}
	keep = []
	for i in range(len(bounding_boxes)):
		if i in avail_idx:
			highest_iou = 0
			highest_iou_box = None
			for j in range(len(bounding_boxes)):
				if j in avail_idx:
					IoU = calc_IOU(np.copy(img), bounding_boxes[i], bounding_boxes[j])
					if IoU == 0:
						continue
					elif IoU > highest_iou:
						highest_iou = IoU
						highest_iou_box = bounding_boxes[i]
						if j in avail_idx:
							avail_idx.remove(j)
			if np.any(keep == highest_iou_box):
				continue
			else:
				keep.append(highest_iou_box)
				if i in avail_idx:
					avail_idx.remove(i)
	
	cropped_images = {}
	sub_image_count = 0
	for region in keep:
		pad = 5
		region[0] -= pad
		region[1] -= pad
		region[2] += pad
		region[3] += pad
		cropped = np.copy(img)[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
		resized_cropped = cv2.resize(np.copy(cropped), (64, 32), None, 0, 0, cv2.INTER_CUBIC)
		resized_cropped = cv2.GaussianBlur(resized_cropped, ksize=(3, 3), sigmaX=1, sigmaY=1,
		                                   borderType=cv2.BORDER_REFLECT_101)
		complete_prediction_vgg = vgg_model.predict(np.asarray([resized_cropped]))
		complete_prediction_lenet = lenet_model.predict(np.asarray([resized_cropped]))
		vgg_pred = np.argmax(complete_prediction_vgg, axis=1)
		lenet_pred = np.argmax(complete_prediction_lenet, axis=1)
		
		cropped_images[f"img_{sub_image_count}"] = {"region": region, "cropped image": cropped,
		                                            "Complete LeNet Prediction": complete_prediction_lenet,
		                                            "LeNet Prediction": lenet_pred,
		                                            "Complete VGG Prediction": complete_prediction_vgg,
		                                            "VGG Prediction": vgg_pred}
		sub_image_count += 1
	
	resulting_image = np.copy(img)
	for key, val in cropped_images.items():
		pt_1 = (val["region"][0], val["region"][1])
		pt_2 = val["region"][:2] + val["region"][2:]
		cv2.rectangle(resulting_image, pt1=tuple(pt_1), pt2=tuple(pt_2), color=(0, 225, 0), thickness=1)
		cv2.putText(resulting_image, text=f"{val['VGG Prediction'][0]}",
		            org=(pt_2[0]-20, pt_2[1]+30),
		            fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 245, 0), thickness=1)
	
	cv2.imwrite(image_2_save_file_name, resulting_image)
	return


def process_image_three():
	# Different Locations
	reshape_size = (32, 64, 3)
	input_file_name = 'img3.png'
	image_3_save_file_name = "3.png"
	img = cv2.imread(input_file_name)
	
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('delta', 'image', 4, 255, nothing)
	# cv2.createTrackbar('min_area', 'image', 783, 1000, nothing)
	# cv2.createTrackbar('max_area', 'image', 1588, 20000, nothing)
	# cv2.createTrackbar('max_variation', 'image', 8, 100, nothing)
	# cv2.createTrackbar('edge_blur_size', 'image', 5, 30, nothing)
	#
	# while (1):
	# 	temp_image = np.copy(img)
	# 	gray = cv2.cvtColor(np.copy(temp_image), cv2.COLOR_BGR2GRAY)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	delta = cv2.getTrackbarPos('delta', 'image')
	# 	min_area = cv2.getTrackbarPos('min_area', 'image')
	# 	max_area = cv2.getTrackbarPos('max_area', 'image')
	# 	max_variation = cv2.getTrackbarPos('max_variation', 'image')
	# 	edge_blur_size = cv2.getTrackbarPos('edge_blur_size', 'image')
	# 	max_variation /= 100.0
	#
	# 	vis = np.copy(temp_image)
	# 	if edge_blur_size % 2 == 1 and edge_blur_size > 0:
	# 		mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
	# 		                       _edge_blur_size=edge_blur_size)
	# 		regions, _ = mser.detectRegions(gray)
	# 		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	# 		cv2.polylines(vis, hulls, 1, (0, 255, 0))
	# 	cv2.imshow('image', vis)
	# cv2.destroyAllWindows()
	
	vgg_model = load_vgg()
	lenet_model = load_lenet()
	mser = cv2.MSER_create(_delta=4, _min_area=783, _max_area=1588, _max_variation=0.08, _edge_blur_size=5)
	# mser = cv2.MSER_create()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	regions, bounding_boxes = mser.detectRegions(gray)
	
	avail_idx = {t for t in range(bounding_boxes.shape[0])}
	keep = []
	for i in range(len(bounding_boxes)):
		if i in avail_idx:
			highest_iou = 0
			highest_iou_box = None
			for j in range(len(bounding_boxes)):
				if j in avail_idx:
					IoU = calc_IOU(np.copy(img), bounding_boxes[i], bounding_boxes[j])
					if IoU == 0:
						continue
					elif IoU > highest_iou:
						highest_iou = IoU
						highest_iou_box = bounding_boxes[i]
						if j in avail_idx:
							avail_idx.remove(j)
			if np.any(keep == highest_iou_box):
				continue
			else:
				keep.append(highest_iou_box)
				if i in avail_idx:
					avail_idx.remove(i)
	
	cropped_images = {}
	sub_image_count = 0
	for region in keep:
		pad = 5
		region[0] -= pad
		region[1] -= pad
		region[2] += pad
		region[3] += pad
		cropped = np.copy(img)[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
		resized_cropped = cv2.resize(np.copy(cropped), (64, 32), None, 0, 0, cv2.INTER_CUBIC)
		resized_cropped = cv2.GaussianBlur(resized_cropped, ksize=(3, 3), sigmaX=1, sigmaY=1,
		                                   borderType=cv2.BORDER_REFLECT_101)
		complete_prediction_vgg = vgg_model.predict(np.asarray([resized_cropped]))
		complete_prediction_lenet = lenet_model.predict(np.asarray([resized_cropped]))
		vgg_pred = np.argmax(complete_prediction_vgg, axis=1)
		lenet_pred = np.argmax(complete_prediction_lenet, axis=1)
		
		cropped_images[f"img_{sub_image_count}"] = {"region": region, "cropped image": cropped,
		                                            "Complete LeNet Prediction": complete_prediction_lenet,
		                                            "LeNet Prediction": lenet_pred,
		                                            "Complete VGG Prediction": complete_prediction_vgg,
		                                            "VGG Prediction": vgg_pred}
		sub_image_count += 1
	
	resulting_image = np.copy(img)
	for key, val in cropped_images.items():
		pt_1 = (val["region"][0], val["region"][1])
		pt_2 = val["region"][:2] + val["region"][2:]
		cv2.rectangle(resulting_image, pt1=tuple(pt_1), pt2=tuple(pt_2), color=(0, 225, 0), thickness=1)
		cv2.putText(resulting_image, text=f"{val['VGG Prediction'][0]}",
		            org=((pt_2[0])-20, int((pt_1[1] + pt_2[1]) / 2)-45),
		            fontScale=0.75, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 245, 0), thickness=1)
	
	cv2.imwrite(image_3_save_file_name, resulting_image)
	return


def process_image_four():
	# Different Lighting Conditions
	reshape_size = (32, 64, 3)
	input_file_name = 'img4.png'
	image_4_save_file_name = "4.png"
	img = cv2.imread(input_file_name)

	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('delta', 'image', 3, 255, nothing)
	# cv2.createTrackbar('min_area', 'image', 783, 1000, nothing)
	# cv2.createTrackbar('max_area', 'image', 1588, 20000, nothing)
	# cv2.createTrackbar('max_variation', 'image', 8, 100, nothing)
	# cv2.createTrackbar('edge_blur_size', 'image', 5, 30, nothing)
	#
	# while (1):
	# 	temp_image = np.copy(img)
	# 	gray = cv2.cvtColor(np.copy(temp_image), cv2.COLOR_BGR2GRAY)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	delta = cv2.getTrackbarPos('delta', 'image')
	# 	min_area = cv2.getTrackbarPos('min_area', 'image')
	# 	max_area = cv2.getTrackbarPos('max_area', 'image')
	# 	max_variation = cv2.getTrackbarPos('max_variation', 'image')
	# 	edge_blur_size = cv2.getTrackbarPos('edge_blur_size', 'image')
	# 	max_variation /= 100.0
	#
	# 	vis = np.copy(temp_image)
	# 	if edge_blur_size % 2 == 1 and edge_blur_size > 0:
	# 		mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
	# 		                       _edge_blur_size=edge_blur_size)
	# 		regions, _ = mser.detectRegions(gray)
	# 		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	# 		cv2.polylines(vis, hulls, 1, (0, 255, 0))
	# 	cv2.imshow('image', vis)
	# cv2.destroyAllWindows()
	
	img = cv2.imread(input_file_name)
	vgg_model = load_vgg()
	lenet_model = load_lenet()
	
	mser = cv2.MSER_create(_delta=25, _min_area=255, _max_area=388, _max_variation=0.64, _edge_blur_size=5)
	# mser = cv2.MSER_create()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	regions, bounding_boxes = mser.detectRegions(gray)
	
	avail_idx = {t for t in range(bounding_boxes.shape[0])}
	keep = set()
	for i in range(len(bounding_boxes)):
		temp = tuple(bounding_boxes[i])
		if temp not in keep:
			keep.add(temp)
	
	cropped_images = {}
	sub_image_count = 0
	for region in keep:
		region = np.asarray(list(region))
		pad = 5
		region[0] -= pad
		region[1] -= pad
		region[2] += pad
		region[3] += pad
		cropped = np.copy(img)[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
		resized_cropped = cv2.resize(np.copy(cropped), (64, 32), None, 0, 0, cv2.INTER_CUBIC)
		resized_cropped = cv2.GaussianBlur(resized_cropped, ksize=(3, 3), sigmaX=1, sigmaY=1,
		                                   borderType=cv2.BORDER_REFLECT_101)
		complete_prediction_vgg = vgg_model.predict(np.asarray([resized_cropped]))
		complete_prediction_lenet = lenet_model.predict(np.asarray([resized_cropped]))
		vgg_pred = np.argmax(complete_prediction_vgg, axis=1)
		lenet_pred = np.argmax(complete_prediction_lenet, axis=1)
		
		cropped_images[f"img_{sub_image_count}"] = {"region": region, "cropped image": cropped,
		                                            "Complete LeNet Prediction": complete_prediction_lenet,
		                                            "LeNet Prediction": lenet_pred,
		                                            "Complete VGG Prediction": complete_prediction_vgg,
		                                            "VGG Prediction": vgg_pred}
		sub_image_count += 1
	
	resulting_image = np.copy(img)
	for key, val in cropped_images.items():
		pt_1 = (val["region"][0], val["region"][1])
		pt_2 = val["region"][:2] + val["region"][2:]
		cv2.rectangle(resulting_image, pt1=tuple(pt_1), pt2=tuple(pt_2), color=(0, 225, 0), thickness=1)
		cv2.putText(resulting_image, text=f"{val['VGG Prediction'][0]}",
		            org=((pt_2[0])-10, int((pt_1[1] + pt_2[1]) / 2)-25),
		            fontScale=0.75, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 245, 0), thickness=1)
	
	cv2.imwrite(image_4_save_file_name, resulting_image)
	return


def process_image_five():
	# Different Scale, Location, and Lighting
	reshape_size = (32, 64, 3)
	input_file_name = 'img5.png'
	image_5_save_file_name = "5.png"
	img = cv2.imread(input_file_name)
	vgg_model = load_vgg()
	lenet_model = load_lenet()

	mser = cv2.MSER_create(_delta=9, _min_area=256, _max_area=2183, _max_variation=0.06, _edge_blur_size=5)
	# mser = cv2.MSER_create()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	regions, bounding_boxes = mser.detectRegions(gray)

	avail_idx = {t for t in range(bounding_boxes.shape[0])}
	keep = []
	for i in range(len(bounding_boxes)):
		if i in avail_idx:
			highest_iou = 0
			highest_iou_box = None
			for j in range(len(bounding_boxes)):
				if j in avail_idx:
					IoU = calc_IOU(np.copy(img), bounding_boxes[i], bounding_boxes[j])
					if IoU == 0:
						continue
					elif IoU > highest_iou:
						highest_iou = IoU
						highest_iou_box = bounding_boxes[i]
						if j in avail_idx:
							avail_idx.remove(j)
			if np.any(keep == highest_iou_box):
				continue
			else:
				keep.append(highest_iou_box)
				if i in avail_idx:
					avail_idx.remove(i)
					
	cropped_images = {}
	sub_image_count = 0
	for region in keep:
		pad = 5
		region[0] -= pad
		region[1] -= pad
		region[2] += pad
		region[3] += pad
		cropped = np.copy(img)[region[1]:region[1]+region[3], region[0]:region[0] + region[2]]
		resized_cropped = cv2.resize(np.copy(cropped), (64, 32), None, 0, 0, cv2.INTER_CUBIC)
		resized_cropped = cv2.GaussianBlur(resized_cropped, ksize=(3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT_101)
		complete_prediction_vgg = vgg_model.predict(np.asarray([resized_cropped]))
		complete_prediction_lenet = lenet_model.predict(np.asarray([resized_cropped]))
		vgg_pred = np.argmax(complete_prediction_vgg, axis=1)
		lenet_pred = np.argmax(complete_prediction_lenet, axis=1)
		
		cropped_images[f"img_{sub_image_count}"] = {"region": region, "cropped image": cropped,
		                                            "Complete LeNet Prediction": complete_prediction_lenet,
		                                            "LeNet Prediction": lenet_pred,
		                                            "Complete VGG Prediction": complete_prediction_vgg,
		                                            "VGG Prediction": vgg_pred}
		sub_image_count += 1
	
	resulting_image = np.copy(img)
	for key, val in cropped_images.items():
		pt_1 = (val["region"][0], val["region"][1])
		pt_2 = val["region"][:2] + val["region"][2:]
		cv2.rectangle(resulting_image, pt1=tuple(pt_1), pt2=tuple(pt_2), color=(0, 225, 0), thickness=1)
		cv2.putText(resulting_image, text=f"{val['VGG Prediction'][0]}", org=((pt_2[0]) + 10, int((pt_1[1] + pt_2[1]) / 2)),
		            fontScale=2, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 245, 0), thickness=1)

	cv2.imwrite(image_5_save_file_name, resulting_image)
	return


if __name__ == "__main__":
	process_image_one()
	process_image_two()
	process_image_three()
	process_image_four()
	process_image_five()
