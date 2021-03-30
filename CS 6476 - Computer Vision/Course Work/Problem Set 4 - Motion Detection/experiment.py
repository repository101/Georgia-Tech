"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0), alternate=False, img=None):
	img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)
	largest = 1e-20
	
	for y in range(0, v.shape[0], stride):
		
		for x in range(0, u.shape[1], stride):
			if alternate:
				
				dist = np.sqrt((u[y, x]) ** 2 + (v[y, x]) ** 2)
				if dist > largest:
					largest = dist
				if dist > 1:
					color = (255, 0, 255)
				else:
					color = (0, 255, 0)
				if color == (255, 0, 255):
					if img is not None:
						cv2.line(img, (x, y), (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), (0, 255, 0), 1)
						cv2.circle(img_out, (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 2, (255, 0, 0), 1)
					else:
						cv2.line(img_out, (x, y), (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), color, 1)
			else:
				try:
					cv2.line(img_out, (x, y), (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), color, 1)
					cv2.circle(img_out, (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 1, color, 1)
				except:
					continue
	if img is not None:
		return img
	else:
		return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
	"""Scales up U and V arrays to match the image dimensions assigned
	to the first pyramid level: pyr[0].

	You will use this method in part 3. In this section you are asked
	to select a level in the gaussian pyramid which contains images
	that are smaller than the one located in pyr[0]. This function
	should take the U and V arrays computed from this lower level and
	expand them to match a the size of pyr[0].

	This function consists of a sequence of ps4.expand_image operations
	based on the pyramid level used to obtain both U and V. Multiply
	the result of expand_image by 2 to scale the vector values. After
	each expand_image operation you should adjust the resulting arrays
	to match the current level shape
	i.e. U.shape == pyr[current_level].shape and
	V.shape == pyr[current_level].shape. In case they don't, adjust
	the U and V arrays by removing the extra rows and columns.

	Hint: create a for loop from level-1 to 0 inclusive.

	Both resulting arrays' shapes should match pyr[0].shape.

	Args:
		u: U array obtained from ps4.optic_flow_lk
		v: V array obtained from ps4.optic_flow_lk
		level: level value used in the gaussian pyramid to obtain U
			   and V (see part_3)
		pyr: gaussian pyramid used to verify the shapes of U and V at
			 each iteration until the level 0 has been met.

	Returns:
		tuple: two-element tuple containing:
			u (numpy.array): scaled U array of shape equal to
							 pyr[0].shape
			v (numpy.array): scaled V array of shape equal to
							 pyr[0].shape
	"""
	# Iterate over the levels starting a level -1 down to -1. The reason for -1 is we still want to process the 0th
	#   level.
	frame = pyr[0]
	for level in range(level - 1, -1, -1):
		current_level = pyr[level]
		u = 2.0 * ps4.expand_image(u)[:current_level.shape[0], :current_level.shape[1]]
		v = 2.0 * ps4.expand_image(v)[:current_level.shape[0], :current_level.shape[1]]
	if u.shape[0] != frame.shape[0] or v.shape[0] != frame.shape[0] or \
			u.shape[1] != frame.shape[1] or v.shape[1] != frame.shape[1]:
		u = u[:frame.shape[0], :frame.shape[1]]
		v = v[:frame.shape[0], :frame.shape[1]]
	return u, v


def part_1a():
	shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
									  'Shift0.png'), 0) / 255.
	shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq',
									   'ShiftR2.png'), 0) / 255.
	shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										  'ShiftR5U5.png'), 0) / 255.
	
	# Optional: smooth the images if LK doesn't work well on raw images
	k_size = 40
	k_type = "uniform"
	sigma = 1
	u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)
	
	# Flow image
	u_v = quiver(u, v, scale=3, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)
	
	# Now let's try with ShiftR5U5. You may want to try smoothing the
	# input images first.
	kernel = np.ones((7, 7)) / 7 ** 2
	
	k_size = 64
	k_type = "uniform"
	sigma = 1
	u, v = ps4.optic_flow_lk(cv2.filter2D(shift_0, -1, kernel=kernel),
							 cv2.filter2D(shift_r5_u5, -1, kernel=kernel), k_size, k_type, sigma)
	
	# Flow image
	u_v = quiver(u, v, scale=3, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)
	return


def part_1b():
	"""Performs the same operations applied in part_1a using the images
	ShiftR10, ShiftR20 and ShiftR40.

	You will compare the base image Shift0.png with the remaining
	images located in the directory TestSeq:
	- ShiftR10.png
	- ShiftR20.png
	- ShiftR40.png

	Make sure you explore different parameters and/or pre-process the
	input images to improve your results.

	In this part you should save the following images:
	- ps4-1-b-1.png
	- ps4-1-b-2.png
	- ps4-1-b-3.png

	Returns:
		None
	"""
	shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
									  'Shift0.png'), 0) / 255.
	shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR10.png'), 0) / 255.
	shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR20.png'), 0) / 255.
	shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR40.png'), 0) / 255.
	
	# Now let's try with shift_r10. You may want to try smoothing the
	# input images first.
	k_size = 56
	k_type = "uniform"
	sigma = 0
	temp_shift_0 = cv2.bilateralFilter(np.copy(shift_0.astype(np.float32)), -1, 6 / 2, 6)
	shift_r10 = cv2.bilateralFilter(shift_r10.astype(np.float32), -1, 6 / 2, 6)
	u, v = ps4.optic_flow_lk(temp_shift_0, shift_r10, k_size, k_type, sigma)
	# Flow image
	u_v = quiver(u, v, scale=3, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)
	
	# Now let's try with shift_r20. You may want to try smoothing the
	# input images first.
	k_size = 88
	k_type = "uniform"
	sigma = 1
	gauss_k_size = 36
	gauss_val_2 = 15
	temp_shift_0 = cv2.blur(np.copy(shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_0 = cv2.blur(np.copy(temp_shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_r20 = cv2.blur(np.copy(shift_r20), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_r20 = cv2.blur(np.copy(temp_shift_r20), (gauss_k_size, gauss_k_size), gauss_val_2)
	u, v = ps4.optic_flow_lk(temp_shift_0, temp_shift_r20, k_size, k_type, sigma)
	
	# Flow image
	u_v = quiver(u, v, scale=0.5, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)
	# def nothing(x):
	#     pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_FREERATIO)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('Gauss Kernel', 'image', 41, 100, nothing)
	# cv2.createTrackbar('Gauss Val 2', 'image', 15, 100, nothing)
	# cv2.createTrackbar('Sigma', 'image', 30, 100, nothing)
	# cv2.createTrackbar('K Size', 'image', 82, 200, nothing)
	#
	# while (1):
	#     k = cv2.waitKey(1) & 0xFF
	#
	#     if k == 27:
	#         break
	#     temp_shift_0 = np.copy(shift_0)
	#     temp_shift_r20 = np.copy(shift_r20)
	#     # get current positions of four trackbars
	#     gauss_k_size = cv2.getTrackbarPos('Gauss Kernel', 'image')
	#     gauss_val_2 = cv2.getTrackbarPos('Gauss Val 2', 'image')
	#     kSize = cv2.getTrackbarPos('K Size', 'image')
	#     sig = cv2.getTrackbarPos('Sigma', 'image')
	#     if gauss_k_size > 0 and kSize > 0 and kSize != 1:
	#         # temp_k = (gauss_k_size * 2) - 1
	#         if gauss_k_size % 2 != 1:
	#             continue
	#         else:
	#             v = 1
	#             # temp_shift_0 = cv2.GaussianBlur(np.copy(shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             # temp_shift_0 = cv2.GaussianBlur(np.copy(temp_shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             # temp_shift_r40 = cv2.GaussianBlur(np.copy(shift_r40), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             # temp_shift_r40 = cv2.GaussianBlur(np.copy(temp_shift_r40), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             temp_shift_0 = cv2.blur(np.copy(temp_shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             temp_shift_0 = cv2.blur(np.copy(temp_shift_0), (gauss_k_size * v, gauss_k_size * v), gauss_val_2 * v)
	#             temp_shift_r20 = cv2.blur(np.copy(temp_shift_r20), (gauss_k_size, gauss_k_size), gauss_val_2)
	#             temp_shift_r20 = cv2.blur(np.copy(temp_shift_r20), (gauss_k_size * v, gauss_k_size * v),
	#                                       gauss_val_2 * v)
	#
	#             # temp_shift_0 = cv2.filter2D(np.copy(shift_0), -1, np.ones((gauss_k_size,gauss_k_size),np.float32)/gauss_k_size**2)
	#             # temp_shift_r40 = cv2.filter2D(np.copy(shift_r40),-1, np.ones((gauss_k_size,gauss_k_size),np.float32)/gauss_k_size**2)
	#
	#             k_type = "uniform"
	#             u, v = ps4.optic_flow_lk(temp_shift_0, temp_shift_r20, kSize, k_type, sigma=sig)
	#
	#             # Flow image
	#             u_v = quiver(u, v, scale=0.5, stride=10, alternate=False)
	#             cv2.imshow('image', u_v)
	#
	# cv2.destroyAllWindows()
	#
	
	# Now let's try with shift_r40. You may want to try smoothing the
	# input images first.
	
	k_size = 82
	k_type = "uniform"
	sigma = 1
	gauss_k_size = 41
	gauss_val_2 = 12
	temp_shift_0 = cv2.blur(np.copy(shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_0 = cv2.blur(np.copy(temp_shift_0), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_r40 = cv2.blur(np.copy(shift_r40), (gauss_k_size, gauss_k_size), gauss_val_2)
	temp_shift_r40 = cv2.blur(np.copy(temp_shift_r40), (gauss_k_size, gauss_k_size), gauss_val_2)
	u, v = ps4.optic_flow_lk(temp_shift_0, temp_shift_r40, k_size, k_type, sigma)
	
	# Flow image
	u_v = quiver(u, v, scale=0.25, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)
	return


def part_2():
	yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
										 'yos_img_01.jpg'), 0) / 255.
	
	# 2a
	levels = 4
	yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
	yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
	cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
				yos_img_01_g_pyr_img)
	
	# 2b
	yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)
	
	yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
	cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
				yos_img_01_l_pyr_img)
	print()


def part_3a_1():
	yos_img_01 = cv2.imread(
		os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
	yos_img_02 = cv2.imread(
		os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
	
	levels = 9  # Define the number of pyramid levels
	yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
	yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
	
	level_id = 3
	k_size = 36
	k_type = "uniform"
	sigma = 3
	u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
							 yos_img_02_g_pyr[level_id],
							 k_size, k_type, sigma)
	
	u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)
	
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)
	
	diff_yos_img = yos_img_01 - yos_img_02_warped
	cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
				ps4.normalize_and_scale(diff_yos_img))
	
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('K Size', 'image', 1, 100, nothing)
	# cv2.createTrackbar('Level Id', 'image', 0, 10, nothing)
	# cv2.createTrackbar('Levels', 'image', 1, 11, nothing)
	#
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('K Size', 'image')
	# 	levelID = cv2.getTrackbarPos('Level Id', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	levels = lvls  # Define the number of pyramid levels
	# 	yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
	# 	yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
	# 	if 0 < lvls < 10 and levelID < lvls and kSize > 0 and kSize % 2 == 1:
	# 		level_id = levelID
	# 		k_size = kSize
	# 		k_type = "uniform"
	# 		sigma = 3
	# 		u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
	# 		                         yos_img_02_g_pyr[level_id],
	# 		                         k_size, k_type, sigma)
	#
	# 		u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)
	#
	# 		interpolation = cv2.INTER_CUBIC  # You may try different values
	# 		border_mode = cv2.BORDER_REFLECT101  # You may try different values
	# 		yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)
	#
	# 		diff_yos_img = yos_img_01 - yos_img_02_warped
	# 		cv2.imshow('image', diff_yos_img)
	#
	# cv2.destroyAllWindows()
	# print()


def part_3a_2():
	yos_img_02 = cv2.imread(
		os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
	yos_img_03 = cv2.imread(
		os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.
	
	levels = 9  # Define the number of pyramid levels
	yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
	yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)
	
	level_id = 3
	k_size = 36
	k_type = "uniform"
	sigma = 3
	u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
							 yos_img_03_g_pyr[level_id],
							 k_size, k_type, sigma)
	
	u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)
	
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)
	
	diff_yos_img = yos_img_02 - yos_img_03_warped
	cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
				ps4.normalize_and_scale(diff_yos_img))
	
	# def nothing(x):
	#     pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_AUTOSIZE)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('R', 'image', value_this_will_default_to, range_of_values, some_function)
	# cv2.createTrackbar('G', 'image', 0, 255, nothing)
	# cv2.createTrackbar('B', 'image', 0, 255, nothing)
	#
	# while (1):
	#     k = cv2.waitKey(1) & 0xFF
	#     if k == 27:
	#         break
	#
	#     # get current positions of four trackbars
	#     r = cv2.getTrackbarPos('R', 'image')
	#     g = cv2.getTrackbarPos('G', 'image')
	#     b = cv2.getTrackbarPos('B', 'image')
	#
	#     levels = 4  # Define the number of pyramid levels
	#     yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
	#     yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)
	#
	#     level_id = 2
	#     k_size = 51
	#     k_type = "uniform"
	#     sigma = 3
	#     u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
	#                              yos_img_03_g_pyr[level_id],
	#                              k_size, k_type, sigma)
	#
	#     u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)
	#
	#     interpolation = cv2.INTER_CUBIC  # You may try different values
	#     border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#     yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)
	#
	#     diff_yos_img = yos_img_02 - yos_img_03_warped
	#
	#     cv2.imshow('image', ps4.normalize_and_scale(diff_yos_img))
	#
	# cv2.destroyAllWindows()


def part_4a():
	shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
									  'Shift0.png'), 0) / 255.
	shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR10.png'), 0) / 255.
	shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR20.png'), 0) / 255.
	shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR40.png'), 0) / 255.
	
	levels = 4
	k_size = 32
	k_type = "uniform"
	sigma = 0
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
								   sigma, interpolation, border_mode)

	u_v = quiver(u10, v10, scale=1, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

	# You may want to try different parameters for the remaining function
	# calls.
	levels = 5
	k_size = 13
	k_type = "uniform"
	sigma = 0
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
								   sigma, interpolation, border_mode)

	u_v = quiver(u20, v20, scale=0.35, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)
	
	levels = 5
	k_size = 41
	k_type = "uniform"
	sigma = 0
	
	u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
								   sigma, interpolation, border_mode)
	u_v = quiver(u40, v40, scale=1, stride=10)
	cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)
	
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 71, 100, nothing)
	# cv2.createTrackbar('Levels', 'image', 4, 10, nothing)
	# # cv2.createTrackbar('B', 'image', 0, 255, nothing)
	#
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	# b = cv2.getTrackbarPos('B', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	# 	k_type = "uniform"
	# 	sigma = 0
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2:
	# 		u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
	# 		                               sigma, interpolation, border_mode)
	#
	# 		u_v = quiver(u20, v20, scale=0.35, stride=10)
	# 		cv2.imshow('image', u_v)
	#
	# cv2.destroyAllWindows()


def part_4b():
	urban_img_01 = cv2.imread(
		os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
	urban_img_02 = cv2.imread(
		os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.
	
	levels = 1
	k_size = 11
	k_type = "uniform"
	sigma = 0
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	
	u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
							   k_type, sigma, interpolation, border_mode)
	
	u_v = quiver(u, v, scale=1, stride=5)
	cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)
	
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 71, 100, nothing)
	# cv2.createTrackbar('Levels', 'image', 4, 10, nothing)
	# cv2.createTrackbar('Sigma', 'image', 1, 100, nothing)
	# cv2.createTrackbar('Quiver Size', 'image', 1, 100, nothing)
	#
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	sig = cv2.getTrackbarPos('Sigma', 'image')
	# 	qSize = cv2.getTrackbarPos('Quiver Size', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	# 	k_type = "gaussian"
	# 	sigma = sig
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2 and qSize > 0 and sig> 0:
	# 		u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
	# 		                           k_type, sigma, interpolation, border_mode)
	#
	# 		u_v = quiver(u, v, scale=qSize/100, stride=10)
	# 		interpolation = cv2.INTER_CUBIC  # You may try different values
	# 		border_mode = cv2.BORDER_REFLECT101  # You may try different values
	# 		urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
	# 		                               border_mode)
	#
	# 		diff_img = urban_img_01 - urban_img_02_warped
	# 		cv2.imshow('image', diff_img)
	#
	# cv2.destroyAllWindows()
	
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
								   border_mode)
	
	diff_img = urban_img_01 - urban_img_02_warped
	cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
				ps4.normalize_and_scale(diff_img))


def part_5a():
	"""Frame interpolation

	Follow the instructions in the problem set instructions.

	Place all your work in this file and this section.
	"""
	shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
									  'Shift0.png'), 0) / 255.
	shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
										'ShiftR10.png'), 0) / 255.
	levels = 4
	k_size = 41
	k_type = "uniform"
	sigma = 0
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	
	u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
								   sigma, interpolation, border_mode)
	images = []

	for i in [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]:
		temp_images = []
		for j in i:
			temp_u = u10 + (u10 * (1.0 - j))
			temp_v = v10 + (v10 * (1.0 - j))
			temp_warped = ps4.warp(np.copy(shift_0), temp_u, temp_v, interpolation, border_mode) * 255.0
			cv2.imwrite(f"Image_{j}.png", temp_warped)
			temp_images.append(temp_warped)
		images.append(temp_images)
	# Create Sequence of interpolated frames 0-1. [0, 0.2, 0.4, 0.6, 0.8, 1.0.]
	# Resulting images should be 2 rows by 3 columns
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 71, 100, nothing)
	# cv2.createTrackbar('Levels', 'image', 4, 10, nothing)
	# # cv2.createTrackbar('scale', 'image', 1, 255, nothing)
	# cv2.createTrackbar('Gauss K Size', 'image', 1, 255, nothing)
	# cv2.createTrackbar('Gauss val 2', 'image', 1, 255, nothing)
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	# scale = cv2.getTrackbarPos('scale', 'image')
	# 	gaussKSize = cv2.getTrackbarPos('Gauss K Size', 'image')
	# 	gauss_val_2 = cv2.getTrackbarPos('Gauss val 2', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	# 	k_type = "uniform"
	# 	sigma = 0
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2 and gaussKSize > 0 and gaussKSize %2 == 1:
	# 		# temp_shift_0 = cv2.blur(np.copy(shift_0), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# temp_shift_r10 = cv2.blur(np.copy(shift_r10), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# u20, v20 = ps4.hierarchical_lk(temp_shift_0, temp_shift_r10, levels, k_size, k_type,
	# 		#                                sigma, interpolation, border_mode)
	# 		u20, v20 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
	# 		                               sigma, interpolation, border_mode)
	#
	# 		u_v = quiver(u20, v20, scale=1.0, stride=10)
	# 		cv2.imshow('image', u_v)
	#
	# cv2.destroyAllWindows()
	
	output_image = np.zeros(shape=(shift_0.shape[0]*2, shift_0.shape[1]*3))
	start_row = 0
	for i in images:
		start_col = 0
		for j in i:
			start_row, end_row = (start_row, j.shape[0] + start_row)
			start_col, end_col = (start_col, j.shape[1] + start_col)
			output_image[start_row:j.shape[0] + start_row, start_col:j.shape[1] + start_col] = j
			start_col += j.shape[1]

		start_row += shift_0.shape[0]
	
	cv2.imwrite("ps4-5-a-1.png", output_image)
	return


def part_5b():
	"""Frame interpolation

	Follow the instructions in the problem set instructions.

	Place all your work in this file and this section.
	"""
	I_01 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
									  'mc01.png'), 0) / 255.
	I_02 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
									  'mc02.png'), 0) / 255.

	images = [I_01]
	levels = 4
	k_size = 41
	interpolation = cv2.INTER_CUBIC
	border_mode = cv2.BORDER_REFLECT101
	u, v = ps4.hierarchical_lk(np.copy(I_01), np.copy(I_02), k_type="uniform", k_size=k_size, levels=levels,
	                           sigma=1, border_mode=border_mode, interpolation=interpolation)
	cv2.imwrite("img_0.png", I_01*255)
	pct = [0.2, 0.4, 0.6, 0.8]
	for i in range(4):
		u, v = ps4.hierarchical_lk(np.copy(images[-1]), np.copy(I_02), k_type="uniform", k_size=k_size, levels=levels,
		                           sigma=1, border_mode=border_mode, interpolation=interpolation)
		temp_u = u
		temp_v = v
		temp_warped = ps4.warp(np.copy(images[-1]), -temp_u, -temp_v, interpolation, border_mode)
		images.append(temp_warped)
	images.append(I_02)
	for i in range(len(images)):
		cv2.imwrite(f"img_{i}.png", images[i]*255)
	output_image = np.zeros(shape=(I_01.shape[0]*2, I_01.shape[1]*3))
	start_col = 0
	start_row = 0
	for i in range(3):
		output_image[start_row:images[i].shape[0] + start_row, start_col:images[i].shape[1] + start_col] = images[i]*255
		start_col += images[i].shape[1]
	start_col = 0
	start_row += images[-1].shape[0]
	for i in range(3):
		output_image[start_row:images[i+3].shape[0] + start_row, start_col:images[i+3].shape[1] + start_col] = images[i+3]*255
		start_col += images[i].shape[1]
	# cv2.imshow("id", output_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite("ps4-5-a-1.png", output_image)

	#
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 1, 100, nothing)
	# cv2.createTrackbar('Levels', 'image', 1, 12, nothing)
	#
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	# 	k_type = "gaussian"
	# 	sigma = 0
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2:
	# 		# temp_shift_0 = cv2.blur(np.copy(I_01), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# temp_shift_r10 = cv2.blur(np.copy(I_02), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# img_01_pyr = ps4.gaussian_pyramid(np.copy(I_01), levels)
	# 		# img_02_pyr = ps4.gaussian_pyramid(np.copy(I_02), levels)
	#
	# 		u, v = ps4.hierarchical_lk(np.copy(I_01), np.copy(I_02), k_type="uniform", k_size=k_size, levels=levels,
	# 		                           sigma=1, border_mode=border_mode, interpolation=interpolation)
	#
	# 		# u, v = scale_u_and_v(u, v, level_id, img_02_pyr)
	#
	# 		u_v = quiver(u, v, scale=3, stride=10)
	# 		temp_warped = ps4.warp(np.copy(I_01), u, v, interpolation, border_mode)
	# 		cv2.imshow('NEW image', (temp_warped * 0.5) + (np.copy(I_01) * 0.5))
	# 		err = np.sum((temp_warped - I_02)**2)
	# 		print(f"Current_Error: {err:.4f}")
	# 		cv2.imshow("Actual", np.copy(I_02))
	# 		cv2.imshow("image", u_v)
	# 		cv2.imshow("just warp", temp_warped)
	#
	# cv2.destroyAllWindows()
	
	
	
	# PART 2
	
	I_03 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
									  'mc03.png'), 0) / 255.
	
	images = [I_02]
	levels = 4
	k_size = 41
	interpolation = cv2.INTER_CUBIC
	border_mode = cv2.BORDER_REFLECT101
	cv2.imwrite("img_0.png", I_01*255)
	pct = [0.2, 0.4, 0.6, 0.8]
	for i in range(4):
		u, v = ps4.hierarchical_lk(np.copy(images[-1]), np.copy(I_03), k_type="uniform", k_size=k_size, levels=levels,
		                           sigma=1, border_mode=border_mode, interpolation=interpolation)
		temp_u = u
		temp_v = v
		temp_warped = ps4.warp(np.copy(images[-1]), -temp_u, -temp_v, interpolation, border_mode)
		images.append(temp_warped)
	images.append(I_03)
	for i in range(len(images)):
		cv2.imwrite(f"img_ptb_{i}.png", images[i]*255)
	output_image = np.zeros(shape=(I_01.shape[0]*2, I_01.shape[1]*3))
	start_col = 0
	start_row = 0
	for i in range(3):
		output_image[start_row:images[i].shape[0] + start_row, start_col:images[i].shape[1] + start_col] = images[i]*255
		start_col += images[i].shape[1]
	start_col = 0
	start_row += images[-1].shape[0]
	for i in range(3):
		output_image[start_row:images[i+3].shape[0] + start_row, start_col:images[i+3].shape[1] + start_col] = images[i+3]*255
		start_col += images[i].shape[1]
	# def nothing(x):
	# 	pass

	# cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 1, 100, nothing)
	# cv2.createTrackbar('Levels', 'image', 1, 12, nothing)
	#
	# while (1):
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	# 	k_type = "gaussian"
	# 	sigma = 0
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2:
	# 		# temp_shift_0 = cv2.blur(np.copy(I_01), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# temp_shift_r10 = cv2.blur(np.copy(I_02), (gaussKSize, gaussKSize), gauss_val_2)
	# 		# img_01_pyr = ps4.gaussian_pyramid(np.copy(I_01), levels)
	# 		# img_02_pyr = ps4.gaussian_pyramid(np.copy(I_02), levels)
	#
	# 		u, v = ps4.hierarchical_lk(np.copy(I_02), np.copy(I_03), k_type="uniform", k_size=k_size, levels=levels,
	# 		                           sigma=1, border_mode=border_mode, interpolation=interpolation)
	#
	# 		# u, v = scale_u_and_v(u, v, level_id, img_02_pyr)
	#
	# 		u_v = quiver(u, v, scale=3, stride=10)
	# 		temp_warped = ps4.warp(np.copy(I_02), u, v, interpolation, border_mode)
	# 		cv2.imshow('NEW image', (temp_warped * 0.5) + (np.copy(I_02) * 0.5))
	# 		err = np.sum((temp_warped - I_03)**2)
	# 		print(f"Current_Error: {err:.4f}")
	# 		cv2.imshow("Actual", np.copy(I_03))
	# 		cv2.imshow("image", u_v)
	# 		cv2.imshow("just warp", temp_warped)
	#
	# cv2.destroyAllWindows()
	
	# output_image = np.zeros(shape=(I_02.shape[0] * 2, I_02.shape[1] * 3))
	# start_row = 0
	# for i in images:
	# 	start_col = 0
	# 	for j in i:
	# 		start_row, end_row = (start_row, j.shape[0] + start_row)
	# 		start_col, end_col = (start_col, j.shape[1] + start_col)
	# 		output_image[start_row:j.shape[0] + start_row, start_col:j.shape[1] + start_col] = j
	# 		start_col += j.shape[1]
	#
	# 	start_row += images[-1].shape[0]
	
	cv2.imwrite("ps4-5-a-2.png", output_image)


def part_6():
	"""Challenge Problem

	Follow the instructions in the problem set instructions.

	Place all your work in this file and this section.
	"""
	interpolation = cv2.INTER_CUBIC  # You may try different values
	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter('ps4-my-video.mp4', fourcc, 20.0, (960, 540))
	cap = cv2.VideoCapture("fortnite_Trim.mp4")
	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Error opening video stream or file")
	i = 0
	# Read until video is completed
	prev = None
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:

			# Display the resulting frame
			# cv2.imshow('Frame', frame)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
			if prev is not None:
				img1 = cv2.cvtColor(cv2.blur(prev, (3, 3)), cv2.COLOR_BGR2GRAY)/255.
				img2 = cv2.cvtColor(cv2.blur(frame, (3,3)), cv2.COLOR_BGR2GRAY)/255.

				u, v = ps4.hierarchical_lk(np.copy(img1), np.copy(img2), k_type="uniform", k_size=48, levels=2,
				                           sigma=1, border_mode=border_mode, interpolation=interpolation)

				# u, v = scale_u_and_v(u, v, level_id, img_02_pyr)

				u_v = quiver(u, v, scale=1, stride=10, alternate=True, img=np.copy(frame))
				# temp_warped = ps4.warp(np.copy(img1), u, v, interpolation, border_mode)
				# cv2.imshow('NEW image', (temp_warped * 0.5) + (np.copy(img1) * 0.5))
				# err = np.sum((temp_warped - img2)**2)
				# print(f"Current_Error: {err:.4f}")
				# cv2.imshow("Actual", np.copy(img2))
				# cv2.imshow("image", u_v)

				cv2.imwrite(f"video_clip_frames/new_frame_{i}.png", u_v)
				out.write(u_v)
				i += 1
				# Press Q on keyboard to  exit
				if cv2.waitKey(1) == ord('q'):
					break
			else:
				prev = np.copy(frame)

		# Break the loop
		else:
			break

	# When everything done, release the video capture object
	cap.release()
	out.release()

	# Closes all the frames
	cv2.destroyAllWindows()
	
	img1_color = cv2.imread("video_clip_frames/frame_48.png", 1)
	img2_color = cv2.imread("video_clip_frames/frame_49.png", 1)
	img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)/255.
	img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)/255.
	# cv2.imshow("t", img1)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	# def nothing(x):
	# 	pass
	#
	# cv2.namedWindow('image', flags=cv2.WINDOW_GUI_NORMAL)
	#
	# # create trackbars for color change
	#
	# cv2.createTrackbar('kSize', 'image', 105, 200, nothing)
	# cv2.createTrackbar('Levels', 'image', 5, 30, nothing)
	#
	# while (1):
	# 	temp_color = np.copy(img1_color)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == 27:
	# 		break
	#
	# 	# get current positions of four trackbars
	# 	kSize = cv2.getTrackbarPos('kSize', 'image')
	# 	lvls = cv2.getTrackbarPos('Levels', 'image')
	# 	levels = lvls
	# 	k_size = kSize
	#
	# 	interpolation = cv2.INTER_CUBIC  # You may try different values
	# 	border_mode = cv2.BORDER_REFLECT101  # You may try different values
	#
	# 	if kSize > 0 and kSize % 2 == 1 and lvls > 0 and kSize > lvls * 2:
	# 		img1 = cv2.blur(img1, (3, 3))
	# 		img2 = cv2.blur(img2, (3, 3))
	#
	# 		u, v = ps4.hierarchical_lk(np.copy(img1), np.copy(img2), k_type="uniform", k_size=kSize, levels=lvls,
	# 		                           sigma=1, border_mode=border_mode, interpolation=interpolation)
	#
	# 		# u, v = scale_u_and_v(u, v, level_id, img_02_pyr)
	#
	# 		u_v = quiver(u, v, scale=1, stride=10, alternate=True, img=np.copy(temp_color))
	# 		# temp_warped = ps4.warp(np.copy(img1), u, v, interpolation, border_mode)
	# 		# cv2.imshow('NEW image', (temp_warped * 0.5) + (np.copy(img1) * 0.5))
	# 		# err = np.sum((temp_warped - img2)**2)
	# 		# print(f"Current_Error: {err:.4f}")
	# 		# cv2.imshow("Actual", np.copy(img2))
	# 		cv2.imshow("image", u_v)
	# 		# cv2.imshow("just warp", temp_warped)
	#
	# cv2.destroyAllWindows()
	


if __name__ == '__main__':
	part_1a()
	part_1b()
	part_2()
	part_3a_1()
	part_3a_2()
	part_4a()
	part_4b()
	part_5a()
	part_5b()
	part_6()
