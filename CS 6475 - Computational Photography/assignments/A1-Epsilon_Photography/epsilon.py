#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import numpy as np
import scipy
import math

from scipy import ndimage

if __name__ == "__main__":
	red_light = cv2.imread("image1.JPG")
	yellow_light = cv2.imread("image2.JPG")
	green_light = cv2.imread("image3.JPG")
	bounding_box = {"x1": 2225, "x2": 3901, "y1": 1357, "y2": 1861}
	
	red_light_focus = red_light[bounding_box["y1"]:bounding_box["y2"], bounding_box["x1"]:bounding_box["x2"]]
	cv2.imwrite("image1_focus.jpg", red_light_focus)
	print("image1_focus.jpg saved\n")
	red_light_focus_grayscale = cv2.cvtColor(red_light_focus, cv2.COLOR_BGR2GRAY)
	
	yellow_light_focus = yellow_light[bounding_box["y1"]:bounding_box["y2"], bounding_box["x1"]:bounding_box["x2"]]
	cv2.imwrite("image2_focus.jpg", yellow_light_focus)
	print("image2_focus.jpg saved\n")
	yellow_light_focus_grayscale = cv2.cvtColor(yellow_light_focus, cv2.COLOR_BGR2GRAY)
	
	green_light_focus = green_light[bounding_box["y1"]:bounding_box["y2"], bounding_box["x1"]:bounding_box["x2"]]
	cv2.imwrite("image3_focus.jpg", green_light_focus)
	print("image3_focus.jpg saved\n")
	green_light_focus_grayscale = cv2.cvtColor(green_light_focus, cv2.COLOR_BGR2GRAY)
	
	result = red_light_focus.copy()
	
	cool_kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
	kernel = np.asarray([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
	default_kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
	vertical_edge_kernel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	horizontal_edge_kernel = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	
	red_light_channel_zero = ndimage.convolve(red_light_focus[1:-1, 1:-1, 0], weights=kernel)
	red_light_channel_one = ndimage.convolve(red_light_focus[1:-1, 1:-1, 1], weights=kernel)
	red_light_channel_two = ndimage.convolve(red_light_focus[1:-1, 1:-1, 2], weights=kernel)
	
	yellow_light_channel_zero = ndimage.convolve(yellow_light_focus[1:-1, 1:-1, 0], weights=kernel)
	yellow_light_channel_one = ndimage.convolve(yellow_light_focus[1:-1, 1:-1, 1], weights=kernel)
	yellow_light_channel_two = ndimage.convolve(yellow_light_focus[1:-1, 1:-1, 2], weights=kernel)
	
	green_light_channel_zero = ndimage.convolve(green_light_focus[1:-1, 1:-1, 0], weights=kernel)
	green_light_channel_one = ndimage.convolve(green_light_focus[1:-1, 1:-1, 1], weights=kernel)
	green_light_channel_two = ndimage.convolve(green_light_focus[1:-1, 1:-1, 2], weights=kernel)
	
	print("Convolution Complete \n")
	
	for column in range(result.shape[0] - 4):
		# use column +1
		for row in range(result.shape[1] - 4):
			temp1 = ((yellow_light_channel_zero[column + 1][row + 1] / 3) +
			         (green_light_channel_zero[column + 1][row + 1] / 3) +
			         (red_light_channel_zero[column + 1][row + 1] / 3))
			
			temp2 = ((yellow_light_channel_one[column + 1][row + 1] / 3) +
			         (green_light_channel_one[column + 1][row + 1] / 3) +
			         (red_light_channel_one[column + 1][row + 1] / 3))
			
			temp3 = ((yellow_light_channel_two[column + 1][row + 1] / 3) +
			         (green_light_channel_two[column + 1][row + 1] / 3) +
			         (red_light_channel_two[column + 1][row + 1] / 3))
			result[column + 1][row + 1] = np.asarray([temp1, temp2, temp3])
		# use row +1
	#         result[column+1][row+1] = GetCrossCorrelationArray(column+1, row+1, red_light_focus,
	#                                                            yellow_light_focus, green_light_focus)[1][1]
	
	cv2.imwrite("final_artifact.jpg", result)
	print("result.jpg saved \n")
	print("Finished")
	
	result1 = ndimage.convolve(red_light_focus_grayscale, weights=default_kernel)
	vertical_edge = ndimage.convolve(red_light_focus_grayscale, weights=vertical_edge_kernel)
	horizontal_edge = ndimage.convolve(red_light_focus_grayscale, weights=horizontal_edge_kernel)
	
	# test = cv2.add(red_light_focus, yellow_light_focus)
	# test2 = cv2.add(test, green_light_focus)
	# red_light[bounding_box["y1"]:bounding_box["y2"], bounding_box["x1"]:bounding_box["x2"]] = test2
	# cv2.imwrite("FINAL.jpg", red_light)
	# cv2.imshow("test2", test2)
	cv2.imshow("Image3 Focus", green_light_focus)
	cv2.imshow("Image1 Focus", red_light_focus)
	cv2.imshow("Image2 Focus", yellow_light_focus)
	cv2.imshow("Results", result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	print("Finsihed")


def CalcAvg(pixel1, pixel2, pixel3):
	return np.asarray([(pixel1[0] + pixel2[0] + pixel3[0]) / 3,
	                   (pixel1[1] + pixel2[1] + pixel3[1]) / 3,
	                   (pixel1[2] + pixel2[2] + pixel3[2]) / 3])


def GetCrossCorrelationArray(x, y, array1, array2, array3):
	a = [array1[x - 1][y - 1], array2[x - 1][y - 1], array3[x - 1][y - 1]]
	b = [array1[x][y - 1], array2[x][y - 1], array3[x][y - 1]]
	c = [array1[x + 1][y - 1], array2[x + 1][y - 1], array3[x + 1][y - 1]]
	d = [array1[x - 1][y], array2[x - 1][y], array3[x - 1][y]]
	e = [array1[x][y], array2[x][y], array3[x][y]]
	f = [array1[x + 1][y], array2[x + 1][y], array3[x + 1][y]]
	g = [array1[x - 1][y + 1], array2[x - 1][y + 1], array3[x - 1][y + 1]]
	h = [array1[x][y + 1], array2[x][y + 1], array3[x][y + 1]]
	i = [array1[x + 1][y + 1], array2[x + 1][y + 1], array3[x + 1][y + 1]]
	
	new_a = CalcAvg(a[0], a[1], a[2])
	new_b = CalcAvg(b[0], b[1], b[2])
	new_c = CalcAvg(c[0], c[1], c[2])
	new_d = CalcAvg(d[0], d[1], d[2])
	new_e = CalcAvg(e[0], e[1], e[2])
	new_f = CalcAvg(f[0], f[1], f[2])
	new_g = CalcAvg(g[0], g[1], g[2])
	new_h = CalcAvg(h[0], h[1], h[2])
	new_i = CalcAvg(i[0], i[1], i[2])
	result = np.asarray([[new_a, new_b, new_c], [new_d, new_e, new_f], [new_g, new_h, new_i]])
	
	test_green = green_light_focus.copy()
	test_yellow = yellow_light_focus.copy()
	test_red = red_light_focus.copy()
	
	green_mask = np.logical_and((test_green[:, :, 0] > 50), (test_green[:, :, 1] > 60), (test_green[:, :, 2] < 20))
	
	yellow_mask = np.logical_and((test_yellow[:, :, 0] < 150), (test_yellow[:, :, 1] > 50),
	                             (test_yellow[:, :, 2] > 100))
	
	red_mask = np.logical_and((test_red[:, :, 0] < 20), (test_red[:, :, 1] > 20), (test_red[:, :, 2] > 80))
	
	test_green[green_mask] = 0
	test_yellow[yellow_mask] = 0
	test_red[red_mask] = 0
	print()
	
	test = test_red.copy()
	test[red_mask] = test_yellow[yellow_mask]
	cv2.imshow('TESTER', test)
	cv2.imshow('Green Test', test_green)
	cv2.imshow('Yellow Test', test_yellow)
	cv2.imshow('Red Test', test_red)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return result


# tpm = cv2.addWeighted(red_light_focus, 1, yellow_light_focus, 1, 0)
# ts = red_light_focus.copy()
# ts = red_light_focus[red_light_focus != yellow_light_focus]
# print("ts.shape: ", ts.shape)
# red1 = red_light_focus.copy()
# print("red_light_flatten_shape: ", red_light_focus.flatten().shape)
# # ts.reshape(red_light_focus.shape[0], red_light_focus[1], red_light_focus[2])
# print(red_light_focus.shape)
# print(red_light_focus.shape[0])
# print(red_light_focus.shape[1])
# print(red_light_focus.shape[2])
#
# tester = np.argwhere(red_light_focus_grayscale != green_light_focus_grayscale)
# print(tester)
# red_light_focus[tester[1], tester[0]] = 0
# cv2.imshow("newred", red_light_focus)
#
# test = cv2.add(red_light_focus, yellow_light_focus)
# test2 = cv2.add(test, green_light_focus)


def GetTestBoxes():
	green_light_bounding_box = {"x1": 1300, "x2": 1400, "y1": 330, "y2": 430}
	red_light_bounding_box = {"x1": 1375, "x2": 1475, "y1": 45, "y2": 145}
	yellow_light_bounding_box = {"x1": 1300, "x2": 1400, "y1": 190, "y2": 290}
	red_focus = cv2.imread("red_light_focus.JPG")
	yellow_focus = cv2.imread("yellow_light_focus.JPG")
	green_focus = cv2.imread("green_light_focus.JPG")
	cv2.imwrite("green_test.jpg", green_focus[green_light_bounding_box["y1"]:green_light_bounding_box["y2"],
	                              green_light_bounding_box["x1"]:green_light_bounding_box["x2"]])
	cv2.imwrite("yellow_test.jpg", yellow_focus[yellow_light_bounding_box["y1"]:yellow_light_bounding_box["y2"],
	                               yellow_light_bounding_box["x1"]:yellow_light_bounding_box["x2"]])
	cv2.imwrite("red_test.jpg", red_focus[red_light_bounding_box["y1"]:red_light_bounding_box["y2"],
	                            red_light_bounding_box["x1"]:red_light_bounding_box["x2"]])
	
	print(np.mean(green_focus[:, :, 0]))
	print(np.mean(green_focus[:, :, 1]))
	print(np.mean(green_focus[:, :, 2]))
	print(np.mean(red_focus[:, :, 0]))
	print(np.mean(red_focus[:, :, 1]))
	print(np.mean(red_focus[:, :, 2]))
	print(np.mean(yellow_focus[:, :, 0]))
	print(np.mean(yellow_focus[:, :, 1]))
	print(np.mean(yellow_focus[:, :, 2]))
	
	print()
	tes = cv2.imread('tes.jpg')
	print("tes")
	print(np.mean(tes))
	print(np.mean(tes[:, :]))
	print(np.mean(tes[:, :, 0]))
	print(np.mean(tes[:, :, 1]))
	print(np.mean(tes[:, :, 2]))
	
	print()
	tes1 = cv2.imread('tes1.jpg')
	print("tes1")
	print(np.mean(tes1))
	print(np.mean(tes1[:, :]))
	print(np.mean(tes1[:, :, 0]))
	print(np.mean(tes1[:, :, 1]))
	print(np.mean(tes1[:, :, 2]))
	
	print()
	tes2 = cv2.imread('tes2.jpg')
	print("tes2")
	print(np.mean(tes2))
	print(np.mean(tes2[:, :]))
	print(np.mean(tes2[:, :, 0]))
	print(np.mean(tes2[:, :, 1]))
	print(np.mean(tes2[:, :, 2]))
	return
