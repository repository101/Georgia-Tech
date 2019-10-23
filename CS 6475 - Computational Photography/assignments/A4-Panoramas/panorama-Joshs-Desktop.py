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
		color_image = False
		if len(image.shape) > 2:
			color_image = True

		corners = np.zeros((4, 1, 2), dtype=np.float32)
		top_left = image[0, 0][:2]
		top_right = image[0, -1][:2]
		cv2.perspectiveTransform()
		bottom_left = image[-1, 0][:2]
		bottom_right = image[-1, -1][:2]

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
		feat_detector = cv2.ORB_create(nfeatures=500)
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
		image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
		image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
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
		print("ha")
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
		canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
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
		kp1, kp2, matches = findMatchesBetweenImages(
			image_1, image_2, num_matches)
		homography = findHomography(kp1, kp2, matches)
		corners_1 = getImageCorners(image_1)
		corners_2 = getImageCorners(image_2)
		min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
		output_image = warpCanvas(image_1, homography, min_xy, max_xy)
		# WRITE YOUR CODE HERE - REPLACE THIS WITH YOUR BLENDING CODE
		min_xy = min_xy.astype(np.int)
		output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
					 -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2
	
		return output_image
	except Exception as BlendImagePairException:
		print("Exception while processing 'blendImagePair'. \n", BlendImagePairException)
	# END OF FUNCTION


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
		print("Exception during mask pyramid.", MaskException)


def getGaussPyramid(img, levels):
	try:
		result_pyramid = [img]
		for i in range(levels):
			gauss_temp = cv2.pyrDown(result_pyramid[i])
			result_pyramid.append(gauss_temp)
		return result_pyramid
	except Exception as GaussException:
		print("Exception when gauss pyramid.", GaussException)


def getLaplacianPyramid(img, levels):
	try:
		result_pyramid = [img]
		for i in range(levels):
			laplacian_temp = cv2.pyrUp(result_pyramid[i])
			result_pyramid.append(laplacian_temp)
		return result_pyramid
	except Exception as LaplacianException:
		print("Exception when laplacian pyramid.", LaplacianException)
	
	
def evenShape(img):
	try:
		height, width = img.shape
		if height % 2 != 0:
			return False
		if width % 2 != 0:
			return False
		return True
	except Exception as err:
		print("Exception when checking shape.", err)

	
def reshapeImage(in_img, out_image):
	try:
		new_in_image = cv2.resize(in_img, (out_image.shape[1], out_image.shape[0]))
		return new_in_image
	except Exception as err:
		print("Exception when trying to resize image.", err)

	
if __name__ =="__main__":
	print("Hey")
	img1 = cv2.imread("temp.jpg", cv2.IMREAD_COLOR)
	img2 = cv2.imread("2_resized.jpg", cv2.IMREAD_COLOR)
	
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	match_count = 100
	image_1_features, image_2_features, shared_features = findMatchesBetweenImages(img1, img2, match_count)
	
	pts1 = []
	pts2 = []
	for match in shared_features:
		pts1.append(image_1_features[match.queryIdx].pt)
		pts2.append(image_2_features[match.trainIdx].pt)
	pts1 = np.asarray(pts1)
	pts2 = np.asarray(pts2)
	
	gauss = getGaussPyramid(img1, 5)
	laplacian = getLaplacianPyramid(img1, 5)
	
	M_hom, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC)
	
	pano_size = (int(M_hom[0, 2] + img2.shape[1]), max(img1.shape[0], img2.shape[0]))
	img_pano = cv2.warpPerspective(img2, M_hom, pano_size)
	
	test = img_pano.copy()
	
	img_pano[0:img1.shape[0], 0:img1.shape[1], :] = img1
	
	if img1.shape[0] % 2 != 0:
		img1 = cv2.resize(img1, (img1.shape[1], img1.shape[0]-1))
	if img1.shape[1] % 2 != 0:
		img1 = cv2.resize(img1, (img1.shape[1]-1, img1.shape[0]))

	down_result_list = []
	gauss_temp = img1.copy()
	down_result_list.append(gauss_temp)

	test_kernel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])*(1/9)
	
	# Create a mask on line of blend
	
	mask = np.ones(shape=img_pano.shape[:2])
	mask2 = np.full(shape=img_pano.shape[:2], fill_value=1)
	
	mask_kernel_size = 50
	box_kernel = np.ones((mask_kernel_size, mask_kernel_size), np.float32) / (mask_kernel_size ** 2)
	
	mask[0:img1.shape[0], 0:img1.shape[1]] = 100
	mask2[0:img1.shape[0], 0:img1.shape[1]] = 100
	
	blurred_mask = cv2.filter2D(mask, -1, box_kernel)
	blurred_mask2 = cv2.filter2D(mask2, -1, box_kernel)
	
	cv2.imwrite("mask.jpg", mask)
	cv2.imwrite("mask2.jpg", mask2)

	cv2.imwrite("blurred_mask.jpg", blurred_mask)
	cv2.imwrite("blurred_mask2.jpg", blurred_mask2)
	temp = cv2.filter2D(img_pano, -1, kernel=test_kernel)

	# cv2.imshow("Image 1", img1)
	# cv2.imshow("Image 2", img2)
	# cv2.imshow("Panorama", np.uint8(img_pano))
	# cv2.imshow("test", np.uint8(test))
	# cv2.imshow("Laplacian Test", np.uint8(final_image))
	pan1_blue, pan1_green, pan1_red = cv2.split(img_pano)
	test = [pan1_blue, pan1_green, pan1_red]
	pan1_list = []
	pan2_list = []
	
	for i in range(3):
		temp1 = test[i] * blurred_mask
		pan1_list.append(temp1)
		temp2 = test[i] * blurred_mask2
		pan2_list.append(temp2)
	
	try1 = cv2.merge([pan1_list[0], pan1_list[1], pan1_list[2]])
	try2 = cv2.merge([pan2_list[0], pan2_list[1], pan2_list[2]])
	temp2 = img_pano + try1
	
	cv2.imwrite("new pano.jpg", try1)
	cv2.imwrite("new pano 2.jpg", try2)
	cv2.imwrite("TestCombine.jpg", temp2)
	# cv2.imwrite("Laplacian Panorama {} pts.jpg".format(match_count), final_image)
	cv2.imwrite("Temp {} pts.jpg".format(match_count), temp)
	# cv2.imwrite("Test {} pts.jpg".format(match_count), test)
	cv2.imwrite("Test Panorama {} pts.jpg".format(match_count), img_pano)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
