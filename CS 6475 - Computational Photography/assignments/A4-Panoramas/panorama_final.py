""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
	1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
	you are being passed in. Do that on your own if you need to save the images
	but the functions should NOT save the image to file.

	2. DO NOT import any other libraries aside from those that we provide.
	You may not import anything else, and you should be able to complete
	the assignment with the given libraries (and in many cases without them).

	3. DO NOT change the format of this file. You may NOT change function
	type signatures (not even named parameters with defaults). You may add
	additional code to this file at your discretion, however it is your
	responsibility to ensure that the autograder accepts your submission.

	4. This file has only been tested in the provided virtual environment.
	You are responsible for ensuring that your code executes properly in the
	virtual machine environment, and that any changes you make outside the
	areas annotated for student code do not impact your performance on the
	autograder system.
"""
import numpy as np
import scipy as sp
import cv2

"""
ANGLE BETWEEN TWO VECTORS
	(DotProduct) / (vector_magnitude) = cos(theta)
	eg. vect_a = [3,4]
		vect_b = [4,3]
		
		dot_product(vect_a, vect_b) = 24
		
		def vector_magnitude(vec):
			return np.sqrt((vec[0]**2 + vec[1]**2))
			
		vector_magnitude(vect_a) = np.sqrt(3**2+4**2) => np.sqrt(9+16) => np.sqrt(25) = 5
		vector_magnitude(vect_b) = 5
		
		24/(5*5) = 24/25 = 0.96
		
		cos(theta) = 0.96
		arcCos(0.96) = theta =
"""

"""
translation = np.asarray([[1,    0,    Translate_x], 
						  [0,    1,    Translate_y],
						  [0,    0,        1]])

scale = np.asarray([[Scale_x,   0,      0], 
					[   0,   Scale_y,   0],
					[   0,      0,      1]])
					
rotation = np.asarray([[np.cos(theta), (-np.sin(theta)), 0],
					   [np.sin(theta),   np.cos(theta),  0],
					   [     0,                0,        1]])
					   
shear = np.asarray([[   1,   Shear_x, 0],
					[Shear_y,   1,    0],
					[   0,      0,    1]])
					
					
affine (6 degrees of freedom) = [[a, b, c],  | [[x],
								 [d, e, f],  |  [y],
								 [0, 0, 1]]  |  [1]]

		  
Projective Transformation (8 degree of freedom) = [[a, b, c],  | [[x],
												   [d, e, f],  |  [y],
												   [g, h, i]]  |  [w]]
		  
"""


def getImageCorners(image):
	"""Return the x, y coordinates for the four corners bounding the input
	image and return them in the shape expected by the cv2.perspectiveTransform
	function. (The boundaries should completely encompass the image.)

	Parameters
	----------
	image : numpy.ndarray
		Input can be a grayscale or color image

	Returns
	-------
	numpy.ndarray(dtype=np.float32)
		Array of shape (4, 1, 2).  The precision of the output is required
		for compatibility with the cv2.warpPerspective function.

	Notes
	-----
		(1) Review the documentation for cv2.perspectiveTransform (which will
		be used on the output of this function) to see the reason for the
		unintuitive shape of the output array.

		(2) When storing your corners, they must be in (X, Y) order -- keep
		this in mind and make SURE you get it right.
	"""
	try:
		if type(image) != np.ndarray:
			image = np.asarray(image)
		
		rows, cols = image.shape[:2]
		
		pt1 = np.asarray([0, 0])  # Top Left
		pt2 = np.asarray([cols, rows])  # Top Right
		pt3 = np.asarray([0, rows])  # Bottom Left
		pt4 = np.asarray([cols, 0])  # Bottom Right
		corners = np.asarray([pt1, pt2, pt3, pt4], dtype=np.float32)
		
		if corners.shape != (4, 1, 2):
			print()
			corners = np.zeros(shape=(4, 1, 2))
			corners[0] = pt1
			corners[1] = pt2
			corners[2] = pt3
			corners[3] = pt4
		if corners.dtype != np.float32:
			corners = np.float32(corners)
		
		return corners
	except Exception as ImageCornerException:
		print("Exception while running 'getImageCorners'.\n", ImageCornerException)


def findMatchesBetweenImages(image_1, image_2, num_matches):
	"""Return the top list of matches between two input images.

	Parameters
	----------
	image_1 : numpy.ndarray
		The first image (can be a grayscale or color image)

	image_2 : numpy.ndarray
		The second image (can be a grayscale or color image)

	num_matches : int
		The number of keypoint matches to find. If there are not enough,
		return as many matches as you can.

	Returns
	-------
	image_1_kp : list<cv2.KeyPoint>
		A list of keypoint descriptors from image_1

	image_2_kp : list<cv2.KeyPoint>
		A list of keypoint descriptors from image_2

	matches : list<cv2.DMatch>
		A list of the top num_matches matches between the keypoint descriptor
		lists from image_1 and image_2

	Notes
	-----
		(1) You will not be graded for this function.
	"""
	try:
		if type(image_1) != np.ndarray:
			image_1 = np.asarray(image_1)
		if type(image_2) != np.ndarray:
			image_2 = np.asarray(image_2)
		num_matches = int(num_matches)
		
		feat_detector = cv2.ORB_create(nfeatures=500)
		if image_1.dtype != np.uint8:
			image_1 = np.uint8(image_1)
		image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
		image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
		bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
		matches = sorted(bfm.match(image_1_desc, image_2_desc),
		                 key=lambda x: x.distance)[:num_matches]
		return image_1_kp, image_2_kp, matches
	except Exception as FindMatchesBetweenImagesException:
		print("Exception while processing 'findMatchesBetweenImages'. \n", FindMatchesBetweenImagesException)


def findHomography(image_1_kp, image_2_kp, matches):
	"""Returns the homography describing the transformation between the
	keypoints of image 1 and image 2.

		************************************************************
		  Before you start this function, read the documentation
				  for cv2.DMatch, and cv2.findHomography
		************************************************************

	Follow these steps:

		1. Iterate through matches and store the coordinates for each
		   matching keypoint in the corresponding array (e.g., the
		   location of keypoints from image_1_kp should be stored in
		   image_1_points).

			NOTE: Image 1 is your "query" image, and image 2 is your
				  "train" image. Therefore, you index into image_1_kp
				  using `match.queryIdx`, and index into image_2_kp
				  using `match.trainIdx`.

		2. Call cv2.findHomography() and pass in image_1_points and
		   image_2_points, using method=cv2.RANSAC and
		   ransacReprojThreshold=5.0.

		3. cv2.findHomography() returns two values: the homography and
		   a mask. Ignore the mask and return the homography.

	Parameters
	----------
	image_1_kp : list<cv2.KeyPoint>
		A list of keypoint descriptors in the first image

	image_2_kp : list<cv2.KeyPoint>
		A list of keypoint descriptors in the second image

	matches : list<cv2.DMatch>
		A list of matches between the keypoint descriptor lists

	Returns
	-------
	numpy.ndarray(dtype=np.float64)
		A 3x3 array defining a homography transform between image_1 and image_2
	"""
	try:
		image_1_points = []
		image_2_points = []
		for match in matches:
			image_1_points.append(image_1_kp[match.queryIdx].pt)
			image_2_points.append(image_2_kp[match.trainIdx].pt)
		image_1_points = np.asarray(image_1_points)
		image_2_points = np.asarray(image_2_points)
		
		homography_matrix, homography_mask = cv2.findHomography(image_1_points, image_2_points,
		                                                        method=cv2.RANSAC, ransacReprojThreshold=5.0)
		if homography_matrix.dtype != np.float64:
			homography_matrix = np.float64(homography_matrix)
		
		return homography_matrix
	
	except Exception as FindHomographyException:
		print("Exception while processing 'findHomography'. \n", FindHomographyException)


def getBoundingCorners(corners_1, corners_2, homography):
	"""Find the coordinates of the top left corner and bottom right corner of a
	rectangle bounding a canvas large enough to fit both the warped image_1 and
	image_2.

	Given the 8 corner points (the transformed corners of image 1 and the
	corners of image 2), we want to find the bounding rectangle that
	completely contains both images.

	Follow these steps:

		1. Use the homography to transform the perspective of the corners from
		   image 1 (but NOT image 2) to get the location of the warped
		   image corners.

		2. Get the boundaries in each dimension of the enclosing rectangle by
		   finding the minimum x, maximum x, minimum y, and maximum y.

	Parameters
	----------
	corners_1 : numpy.ndarray of shape (4, 1, 2)
		Output from getImageCorners function for image 1

	corners_2 : numpy.ndarray of shape (4, 1, 2)
		Output from getImageCorners function for image 2

	homography : numpy.ndarray(dtype=np.float64)
		A 3x3 array defining a homography transform between image_1 and image_2

	Returns
	-------
	numpy.ndarray(dtype=np.float64)
		2-element array containing (x_min, y_min) -- the coordinates of the
		top left corner of the bounding rectangle of a canvas large enough to
		fit both images (leave them as floats)

	numpy.ndarray(dtype=np.float64)
		2-element array containing (x_max, y_max) -- the coordinates of the
		bottom right corner of the bounding rectangle of a canvas large enough
		to fit both images (leave them as floats)

	Notes
	-----
		(1) The inputs may be either color or grayscale, but they will never
		be mixed; both images will either be color, or both will be grayscale.

		(2) Python functions can return multiple values by listing them
		separated by commas.

		Ex.
			def foo():
				return [], [], []
	"""
	try:
		if type(corners_1) != np.ndarray:
			corners_1 = np.asarray(corners_1)
			if corners_1.shape != (4, 1, 2):
				corners_1 = np.reshape(corners_1, newshape=(4, 1, 2))
		
		if type(corners_2) != np.ndarray:
			corners_2 = np.asarray(corners_2)
			if corners_2.shape != (4, 1, 2):
				corners_2 = np.reshape(corners_2, newshape=(4, 1, 2))
		
		if type(homography) != np.ndarray:
			homography = np.asarray(homography, np.float64)
		if homography.dtype != np.float64:
			homography = np.float64(homography)
		
		transformed_corners_1 = np.vstack(cv2.perspectiveTransform(corners_1, m=homography))
		corners_2_vstack = np.vstack(corners_2)
		min_x = min(np.min(transformed_corners_1[:, 0]), np.min(corners_2_vstack[:, 0]))
		max_x = max(np.max(transformed_corners_1[:, 0]), np.max(corners_2_vstack[:, 0]))
		min_y = min(np.min(transformed_corners_1[:, 1]), np.min(corners_2_vstack[:, 1]))
		max_y = max(np.max(transformed_corners_1[:, 1]), np.max(corners_2_vstack[:, 1]))
		
		min_array = np.asarray([min_x, min_y], dtype=np.float64)
		max_array = np.asarray([max_x, max_y], dtype=np.float64)
		
		return min_array, max_array
	except Exception as GetBoundingCornerException:
		print("Exception when processing 'getBoundingCorners'.\n", GetBoundingCornerException)


def warpCanvas(image, homography, min_xy, max_xy):
	"""Warps the input image according to the homography transform and embeds
	the result into a canvas large enough to fit the next adjacent image
	prior to blending/stitching.

	Follow these steps:

		1. Create a translation matrix (numpy.ndarray) that will shift
		   the image by x_min and y_min. This looks like this:

			[[1, 0, -x_min],
			 [0, 1, -y_min],
			 [0, 0, 1]]

		2. Compute the dot product of your translation matrix and the
		   homography in order to obtain the homography matrix with a
		   translation.

		NOTE: Matrix multiplication (dot product) is not the same thing
			  as the * operator (which performs element-wise multiplication).
			  See Numpy documentation for details.

		3. Call cv2.warpPerspective() and pass in image 1, the combined
		   translation/homography transform matrix, and a vector describing
		   the dimensions of a canvas that will fit both images.

		NOTE: cv2.warpPerspective() is touchy about the type of the output
			  shape argument, which should be an integer.

	Parameters
	----------
	image : numpy.ndarray
		A grayscale or color image (test cases only use uint8 channels)

	homography : numpy.ndarray(dtype=np.float64)
		A 3x3 array defining a homography transform between two sequential
		images in a panorama sequence

	min_xy : numpy.ndarray(dtype=np.float64)
		2x1 array containing the coordinates of the top left corner of a
		canvas large enough to fit the warped input image and the next
		image in a panorama sequence

	max_xy : numpy.ndarray(dtype=np.float64)
		2x1 array containing the coordinates of the bottom right corner of
		a canvas large enough to fit the warped input image and the next
		image in a panorama sequence

	Returns
	-------
	numpy.ndarray(dtype=image.dtype)
		An array containing the warped input image embedded in a canvas
		large enough to join with the next image in the panorama; the output
		type should match the input type (following the convention of
		cv2.warpPerspective)

	Notes
	-----
		(1) You must explain the reason for multiplying x_min and y_min
		by negative 1 in your writeup.
	"""
	# canvas_size properly encodes the size parameter for cv2.warpPerspective,
	# which requires a tuple of ints to specify size, or else it may throw
	# a warning/error, or fail silently
	try:
		if type(image) != np.ndarray:
			image = np.asarray(image)
		
		# region Homography
		if type(homography) != np.ndarray:
			homography = np.asarray(homography, dtype=np.float64)
		if homography.dtype != np.float64:
			homography = np.float64(homography)
		if homography.shape != (3, 3):
			homography = np.reshape(homography, (3, 3))
		# endregion
		
		# region Min_XY
		if type(min_xy) != np.ndarray:
			min_xy = np.asarray(min_xy, dtype=np.float64)
		if min_xy.dtype != np.float64:
			min_xy = np.float64(min_xy)
		
		# endregion
		
		# region Max_XY
		if type(max_xy) != np.ndarray:
			max_xy = np.asarray(max_xy, dtype=np.float64)
		if max_xy.dtype != np.float64:
			max_xy = np.float64(max_xy)
		
		# endregion
		
		translation_matrix = np.asarray([[1, 0, -min_xy[0]],
		                                 [0, 1, -min_xy[1]],
		                                 [0, 0, 1]])
		
		translation_homography_dot_product = np.dot(translation_matrix, homography)
		canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
		panorama = cv2.warpPerspective(image, translation_homography_dot_product, canvas_size)
		displayAndSave(panorama, "WarpPerspective.jpg")
		return panorama
	except Exception as WarpCanvasException:
		print("Exception while processing 'warpCanvas'. \n", WarpCanvasException)


def blendImagePair(image_1, image_2, num_matches):
	"""This function takes two images as input and fits them onto a single
	canvas by performing a homography warp on image_1 so that the keypoints
	in image_1 aligns with the matched keypoints in image_2.

	**************************************************************************

		You MUST replace the basic insertion blend provided here to earn
						 credit for this function.

	   The most common implementation is to use alpha blending to take the
	   average between the images for the pixels that overlap, but you are
					encouraged to use other approaches.

		   Be creative -- good blending is the primary way to earn
				  Above & Beyond credit on this assignment.

	**************************************************************************

	Parameters
	----------
	image_1 : numpy.ndarray
		A grayscale or color image

	image_2 : numpy.ndarray
		A grayscale or color image

	num_matches : int
		The number of keypoint matches to find between the input images

	Returns:
	----------
	numpy.ndarray
		An array containing both input images on a single canvas

	Notes
	-----
		(1) This function is not graded by the autograder. It will be scored
		manually by the TAs.

		(2) The inputs may be either color or grayscale, but they will never be
		mixed; both images will either be color, or both will be grayscale.

		(3) You can modify this function however you see fit -- e.g., change
	"""
	try:
		# Found that too few matches distorts panorama
		img1_b, img1_g, img1_r = cv2.split(image_1)
		img2_b, img2_g, img2_r = cv2.split(image_2)
		img2_b = matchHistogramRoutine(img1_b, img2_b)
		img2_g = matchHistogramRoutine(img1_g, img2_g)
		img2_r = matchHistogramRoutine(img1_r, img2_r)
		test_result = cv2.merge((img2_b.astype(np.uint8), img2_g.astype(np.uint8), img2_r.astype(np.uint8)))
		#
		# getHistograms(image_1)
		# getHistograms(image_2)
		kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2, num_matches)
		displayAndSave(image_1, "image1")
		displayAndSave(image_2, "image2")
		homography = findHomography(kp1, kp2, matches)
		corners_1 = getImageCorners(image_1)
		corners_2 = getImageCorners(image_2)
		min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
		output_image = warpCanvas(image_1, homography, min_xy, max_xy)
		min_xy = min_xy.astype(np.int)

		output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0], -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

		return output_image
	except Exception as BlendImagePairException:
		print("Exception while processing 'blendImagePair'. \n", BlendImagePairException)

#
# def getHistograms(image_hist):
# 	img_copy= image_hist.copy()
# 	if len(image_hist.shape) > 1:
# 		b, g, r = cv2.split(image_hist)
# 		b = cv2.equalizeHist(b)
# 		g = cv2.equalizeHist(g)
# 		r = cv2.equalizeHist(r)
# 		image_hist = cv2.merge((b, g, r))
# 	n=0
# 	color = ('b', 'g', 'r')
# 	for i, col in enumerate(color):
# 		plt.figure(n)
# 		histr = cv2.calcHist([image_hist], [i], None, [256], [0, 256])
# 		histr_2 = cv2.calcHist([img_copy], [i], None, [256], [0, 256])
# 		if i == 0:
# 			title = "Blue Historgram"
# 		if i ==1:
# 			title = "Green Histogram"
# 		if i == 2:
# 			title = "Red Histogram"
# 		plt.title(title)
# 		plt.plot(histr, color=col, label="After Equalized")
# 		plt.plot(histr_2, color='m', label="Before Equalized")
# 		plt.legend()
# 		plt.xlim([0, 256])
# 		plt.savefig("{}.png".format(title))
# 		n += 1
# 	plt.show()
#
#
#
def blendRoutine(min_row_percent, max_row_percent, min_column_percent, max_column_percent, image_a, image_b, levels=4):
	try:
		# Generate Gaussian for Image A
		image_a_gaussian = getGaussPyramid(image_a, levels=levels)
		n = 0
		for i in image_a_gaussian:
			displayAndSave(i, "image_a_{}_gaussian.jpg".format(n))
			n += 1
		# Generate Gaussian for Image B
		image_b_gaussian = getGaussPyramid(image_b, levels=levels)
		n = 0
		for i in image_b_gaussian:
			displayAndSave(i, "image_b_{}_gaussian.jpg".format(n))
			n += 1
		# Generate Laplacian for Image A
		image_a_laplacian = getLaplacianPyramid(image_a, image_a_gaussian, levels=levels)
		n = 0
		for i in image_a_laplacian:
			displayAndSave(i, "image_a_{}_laplacian.jpg".format(n))
			n += 1
		
		# Generate Laplacian for Image B
		image_b_laplacian = getLaplacianPyramid(image_b, image_b_gaussian, levels=levels)
		n = 0
		for i in image_b_laplacian:
			displayAndSave(i, "image_b_{}_laplacian.jpg".format(n))
			n += 1
		
		# Combine parts of both
		combined_laplacian = []
		n = 0
		for lapA, lapB in zip(image_a_laplacian, image_b_laplacian):
			rows, cols = lapA.shape[:2]
			temp = lapA.copy()
			temp_b = lapB.copy()
			row_b, cols_b = temp_b.shape[:2]
			start_row = int(rows * min_row_percent)
			end_row = int(rows * max_row_percent) + start_row
			start_column = int(cols * min_column_percent)
			end_column = int(cols * max_column_percent) + start_column
			if end_row - start_row != row_b:
				end_row += 1
			if end_column - start_column != cols_b:
				end_column += 1
			if end_column > cols:
				end_column = cols
			if end_row > rows:
				end_row = rows
			
			temp[start_row:end_row, start_column:end_column] = lapB
			displayAndSave(temp, "combined_{}.jpg".format(n))
			n += 1
			combined_laplacian.append(temp)
		result_image = combined_laplacian[0]
		for i in range(1, levels + 1):
			result_image = cv2.pyrUp(result_image)
			if result_image.shape != combined_laplacian[i].shape:
				result_image = reshapeImage(result_image, combined_laplacian[i])
			result_image = cv2.add(result_image, combined_laplacian[i])
		
		displayAndSave(result_image, "Pyramid Results.jpg")
		return result_image
	except Exception as RoutineException:
		print("Exception while running blendRoutine.\n", RoutineException)


def getMaskPyramid(mask, up=False, down=False, levels=0):
	try:
		result_pyramid = [mask]
		if up:
			for i in range(levels):
				mask_up_temp = cv2.pyrUp(result_pyramid[i])
				result_pyramid.append(mask_up_temp)
		elif down:
			for i in range(levels):
				mask_down_temp = cv2.pyrDown(result_pyramid[i])
				result_pyramid.append(mask_down_temp)
		else:
			return
		return result_pyramid
	except Exception as MaskException:
		print("Exception occurred while running 'getMaskPyramid'.\n", MaskException)


def getGaussPyramid(img, levels):
	try:
		result_pyramid = [img]
		for i in range(levels):
			gauss_temp = cv2.pyrDown(result_pyramid[i])
			result_pyramid.append(gauss_temp)
		return result_pyramid
	except Exception as GaussException:
		print("Exception occurred while running 'getGaussPyramid'.\n", GaussException)


def getLaplacianPyramid(img, gauss_pyr, levels):
	try:
		result_pyramid = [img]
		for i in range(levels, 0, -1):
			gauss_UP = cv2.pyrUp(gauss_pyr[i])
			if gauss_UP.shape != gauss_pyr[i - 1].shape:
				gauss_UP = reshapeImage(gauss_UP, gauss_pyr[i - 1])
			lapl_level = cv2.subtract(gauss_pyr[i - 1], gauss_UP)
			result_pyramid.append(lapl_level)
		return result_pyramid
	except Exception as LaplacianException:
		print("Exception occurred while running 'getLaplacianPyramid'.\n", LaplacianException)


def evenShape(img):
	try:
		height, width = img.shape
		if height % 2 != 0:
			return False
		if width % 2 != 0:
			return False
		return True
	except Exception as err:
		print("Exception occurred while running 'evenShape'.\n", err)


def reshapeImage(in_img, out_image):
	try:
		new_in_image = cv2.resize(in_img, (out_image.shape[1], out_image.shape[0]))
		return new_in_image
	except Exception as err:
		print("Exception occurred while running 'reshapeImage'.\n", err)


def displayAndSave(img, imgName):
	try:
		if img is None:
			return
		print("Image Name: {}".format(imgName))
		print("Image Shape: {}".format(img.shape))
		print("Image Dtype: {}\n".format(img.dtype))
		split = imgName.split(".")
		if len(split) < 2:
			imgName += ".jpg"
		cv2.imwrite("{}".format(imgName), img)
	except Exception as DisplayAndSaveException:
		print("Exception occurred while running 'displayAndSave'.\n", DisplayAndSaveException)


def getMask(panorama_image, image1, kernel_size, min_xy):
	try:
		# Create a mask on line of blend
		mask = np.zeros(shape=panorama_image.shape[:2])
		
		# rows = -min_xy[1]: -min_xy[1] + image_2.shape[0]
		# cols = -min_xy[0]: -min_xy[0] + image_2.shape[1]
		mask_kernel_size = kernel_size
		box_kernel = np.ones((mask_kernel_size, mask_kernel_size), np.float32) / (mask_kernel_size ** 2)
		
		# Since we want the mask to show 50% in both images we subtract half of kernel size from mask. Only change Columns
		mask_left = mask.copy()
		buffer = kernel_size * 2
		new_mask = mask_left.copy()
		mask_left[-min_xy[1]: -min_xy[1] + image1.shape[0], -min_xy[0]: -min_xy[0] + image1.shape[1]] = 255
		
		new_mask[(-min_xy[1] - buffer): (-min_xy[1] + image1.shape[0] + buffer),
				  (-min_xy[0] - buffer): (-min_xy[0] + image1.shape[1] + buffer)] = 255

		new_mask[(-min_xy[1] - buffer):(-min_xy[1] + image1.shape[0] + buffer):2,
                 (-min_xy[0] - buffer):(-min_xy[0] + buffer):2] = 100

		new_mask[(-min_xy[1] - buffer + 1):(-min_xy[1] + image1.shape[0] + buffer):2,
                 (-min_xy[0] - buffer + 1):(-min_xy[0] + buffer):2] = 100

		mask_right = mask_left.copy()
		mask_right[mask_left == 255] = 0
		mask_right[mask_left == 0] = 255
		
		generated_blur_mask_left = cv2.filter2D(mask_left, -1, box_kernel)
		generated_blur_mask_right = cv2.filter2D(mask_right, -1, box_kernel)
		
		displayAndSave(generated_blur_mask_right, "mask right.jpg")
		displayAndSave(generated_blur_mask_left, "mask left.jpg")
		return generated_blur_mask_left, generated_blur_mask_right,
	except Exception as GetMaskException:
		print("Exception occurred while executing 'getMask'.\n", GetMaskException)


def getPanorama(image_1, image_2, matches_to_find):
	try:
		image_1_features, image_2_features, shared_features = findMatchesBetweenImages(image_1, image_2,
		                                                                               matches_to_find)
		
		pts1 = []
		pts2 = []
		for match in shared_features:
			pts1.append(image_1_features[match.queryIdx].pt)
			pts2.append(image_2_features[match.trainIdx].pt)
		pts1 = np.asarray(pts1)
		pts2 = np.asarray(pts2)
		if len(pts1 > 10):
			M_hom, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC)
			pano_size = (
				int(M_hom[0, 2] + max(image_1.shape[1], image_2.shape[1])), max(image_1.shape[0], image_2.shape[0]))
			img_pano = cv2.warpPerspective(image_2, M_hom, pano_size)
			img_pano[0:image_1.shape[0], 0:image_1.shape[1], :] = image_1
			return img_pano
	except Exception as GetPanoramaException:
		print("Exception occurred while executing 'getMask'.\n", GetPanoramaException)


def testMethod(img1, img2):
	try:
		color = ('b', 'g', 'r')
		n = 0
		img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
		img4 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
		
		cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(51, 51))
		b, g, r = cv2.split(img1)
		b1, g1, r1 = cv2.split(img2)
		b = cl.apply(b)
		g = cl.apply(g)
		r = cl.apply(r)
		b1 = cl.apply(b1)
		g1 = cl.apply(g1)
		r1 = cl.apply(r1)
		
		new_img1 = cv2.merge((b, g, r))
		new_img2 = cv2.merge((b1, g1, r1))
		displayAndSave(new_img1, "New IMG1")
		displayAndSave(new_img2, "New IMG2")
		
		b1, g1, r1 = cv2.split(img1)
		b2, g2, r2 = cv2.split(img2)
		h1, s1, v1 = cv2.split(img3)
		h2, s2, v2 = cv2.split(img4)
		
		v1_avg = np.average(v1)
		v2_avg = np.average(v2)
		v_diff = np.absolute((v1_avg - v2_avg))
		
		h1_avg = np.average(h1)
		h2_avg = np.average(h2)
		h_diff = np.absolute((h1_avg - h2_avg))
		
		s1_avg = np.average(s1)
		s2_avg = np.average(s2)
		s_diff = np.absolute((s1_avg - s2_avg))
		
		displayAndSave(img3, "Image 1 HSV before equalise.jpg")
		displayAndSave(img4, "Image 2 HSV before equalise.jpg")
		s1 = cv2.equalizeHist(s1)
		s2 = cv2.equalizeHist(s2)
		img3 = cv2.merge((h1, s1, v1))
		img4 = cv2.merge((h2, s2, v2))
		img5 = cv2.cvtColor(img3, cv2.COLOR_HSV2BGR)
		img6 = cv2.cvtColor(img4, cv2.COLOR_HSV2BGR)
		
		displayAndSave(img3, "Image 1 HSV after equalise.jpg")
		displayAndSave(img4, "Image 2 HSV after equalise.jpg")
		displayAndSave(img5, "Image 1 HSV2BGR after equalise.jpg")
		displayAndSave(img6, "Image 2 HSV2BGR after equalise.jpg")
		
		b1 = cv2.equalizeHist(b1)
		b2 = cv2.equalizeHist(b2)
		g1 = cv2.equalizeHist(g1)
		g2 = cv2.equalizeHist(g2)
		r1 = cv2.equalizeHist(r1)
		r2 = cv2.equalizeHist(r2)
		img1 = cv2.merge((b1, g1, r1))
		img2 = cv2.merge((b2, g2, r2))
		displayAndSave(img1, "Image_1_Equalized.jpg")
		displayAndSave(img2, "Image_2_Equalized.jpg")
		
		return new_img1, new_img2
	except Exception as TestMethodException:
		print("Exception occurred while executing 'testMethod'.\n", TestMethodException)


def matchHistogramRoutine(histo_1_img, histo_2_img):
	try:
		temp_img1 = histo_1_img.copy()
		temp_img2 = histo_2_img.copy()
		
		img1_val, img1_idx, img1_cts = np.unique(temp_img1.ravel(), return_inverse=True, return_counts=True)
		img2_val, img2_cts = np.unique(temp_img2.ravel(), return_counts=True)
		
		img1_quant = np.cumsum(img1_cts) / temp_img1.size
		img2_quant = np.cumsum(img2_cts) / temp_img2.size
		
		histogram_match_result = np.interp(img1_quant, img2_quant, img2_val)
		histogram_match_result = histogram_match_result[img1_idx].reshape(histo_1_img.shape)
		return histogram_match_result
	
	except Exception as ColorCorrectionException:
		print("Exception while attempting to correct color.\n", ColorCorrectionException)


if __name__ == "__main__":
	img1 = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
	img2 = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
	img3 = cv2.imread("3.jpg", cv2.IMREAD_COLOR)

	images = [img1, img2, img3]
	result = images[0]
	for i in range(len(images)-1):
		result = blendImagePair(result, images[i+1], num_matches=200)
		displayAndSave(result, "Results_{}.jpg".format(i))

	print()
