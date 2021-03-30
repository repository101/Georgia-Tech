"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
	"""Normalizes and scales an image to a given range [0, 255].

	Utility function. There is no need to modify it.

	Args:
		image_in (numpy.array): input image.
		scale_range (tuple): range values (min, max). Default set to [0, 255].

	Returns:
		numpy.array: output image.
	"""
	image_out = np.zeros(image_in.shape)
	cv2.normalize(image_in, image_out, alpha=scale_range[0],
	              beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
	
	return image_out


# Assignment code
def gradient_x(image):
	"""Computes image gradient in X direction.

	Use cv2.Sobel to help you with this function. Additionally you
	should set cv2.Sobel's 'scale' parameter to one eighth and ksize
	to 3.

	Args:
		image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

	Returns:
		numpy.array: image gradient in the X direction. Output
					 from cv2.Sobel.
	"""
	
	return cv2.Sobel(image, -1, dx=1, dy=0, ksize=3, scale=1.0 / 8.0)


def gradient_y(image):
	"""Computes image gradient in Y direction.

	Use cv2.Sobel to help you with this function. Additionally you
	should set cv2.Sobel's 'scale' parameter to one eighth and ksize
	to 3.

	Args:
		image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

	Returns:
		numpy.array: image gradient in the Y direction.
					 Output from cv2.Sobel.
	"""
	
	return cv2.Sobel(image, -1, dx=0, dy=1, ksize=3, scale=1.0 / 8.0)


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
	"""Computes optic flow using the Lucas-Kanade method.

	For efficiency, you should apply a convolution-based method.

	Note: Implement this method using the instructions in the lectures
	and the documentation.

	You are not allowed to use any OpenCV functions that are related
	to Optic Flow.

	Args:
		img_a (numpy.array): grayscale floating-point image with
							 values in [0.0, 1.0].
		img_b (numpy.array): grayscale floating-point image with
							 values in [0.0, 1.0].
		k_size (int): size of averaging kernel to use for weighted
					  averages. Here we assume the kernel window is a
					  square so you will use the same value for both
					  width and height.
		k_type (str): type of kernel to use for weighted averaging,
					  'uniform' or 'gaussian'. By uniform we mean a
					  kernel with the only ones divided by k_size**2.
					  To implement a Gaussian kernel use
					  cv2.getGaussianKernel. The autograder will use
					  'uniform'.
		sigma (float): sigma value if gaussian is chosen. Default
					   value set to 1 because the autograder does not
					   use this parameter.

	Returns:
		tuple: 2-element tuple containing:
			U (numpy.array): raw displacement (in pixels) along
							 X-axis, same size as the input images,
							 floating-point type.
			V (numpy.array): raw displacement (in pixels) along
							 Y-axis, same size and type as U.
	"""
	
	if k_type.lower() == "uniform":
		kernel = np.ones((k_size, k_size)) / (k_size ** 2)
	
	else:
		temp_kernel = cv2.getGaussianKernel(sigma=sigma, ksize=k_size)
		kernel = np.outer(temp_kernel, temp_kernel)
	
	Ix = gradient_x(image=img_a)
	Iy = gradient_y(image=img_a)
	It = img_b - img_a
	
	# https://www.robotics.hiroshima-u.ac.jp/researches/researches_single/823/
	# https://piazza.com/class/kjliq19wrwi2rj?cid=271_f5
	
	Sxx = cv2.filter2D(Ix ** 2, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
	# Sxx[np.where((Sxx == 0) | (Sxx == np.nan))] = 1e-9
	Syy = cv2.filter2D(Iy ** 2, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
	# Syy[np.where((Syy == 0) | (Syy == np.nan))] = 1e-9
	Sxy = cv2.filter2D(Ix * Iy, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
	# Sxy[np.where((Sxy == 0) | (Sxy == np.nan))] = 1e-9
	Sxt = cv2.filter2D(Ix * It, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
	# Sxt[np.where((Sxt == 0) | (Sxt == np.nan))] = 1e-9
	Syt = cv2.filter2D(Iy * It, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
	# Syt[np.where((Syt == 0) | (Syt == np.nan))] = 1e-9
	
	det = (Sxx * Syy - Sxy**2)**-1
	det_check = det[(det == np.nan) | (det == np.inf)]
	
	U_pta = (Sxy * Syt - Syy * Sxt)
	# U_pta[np.where((U_pta == 0) | (U_pta == np.nan))] = 1e-9
	U_ptb = (Sxx * Syy - Sxy ** 2)
	# U_ptb[np.where((U_ptb == 0) | (U_ptb == np.nan))] = 1e-9
	V_pta = (Sxy * Sxt - Sxx * Syt)
	# V_pta[np.where((V_pta == 0) | (V_pta == np.nan))] = 1e-9
	V_ptb = (Sxx * Syy - Sxy ** 2)
	# V_ptb[np.where((V_ptb == 0) | (V_ptb == np.nan))] = 1e-9
	U = U_pta / U_ptb
	V = V_pta / V_ptb
	if det_check.size > 0:
		U[np.where(det == np.nan)] = 0
		U[np.where(det == np.inf)] = 0
		V[np.where(det == np.nan)] = 0
		V[np.where(det == np.inf)] = 0
	# Final check to make sure no nans in array.
	# U[np.where(U == np.nan) | np.where(U == np.inf)] = 0
	# V[np.where(V == np.nan) | np.where(V == np.inf)] = 0

	return U, V


def reduce_image(image):
	"""Reduces an image to half its shape.

	The autograder will pass images with even width and height. It is
	up to you to determine values with odd dimensions. For example the
	output image can be the result of rounding up the division by 2:
	(13, 19) -> (7, 10)

	For simplicity and efficiency, implement a convolution-based
	method using the 5-tap separable filter.

	Follow the process shown in the lecture 6B-L3. Also refer to:
	-  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
	   as a Compact Image Code
	You can find the link in the problem set instructions.

	Args:
		image (numpy.array): grayscale floating-point image, values in
							 [0.0, 1.0].

	Returns:
		numpy.array: output image with half the shape, same type as the
					 input image.
	"""
	# this is the kernel specified in Piazza
	return cv2.filter2D(image, ddepth=-1, kernel=np.outer(np.asarray([1, 4, 6, 4, 1]) / 16.0,
	                                                      np.asarray([1, 4, 6, 4, 1]) / 16.0),
	                    borderType=cv2.BORDER_REFLECT101)[::2, ::2]


def gaussian_pyramid(image, levels):
	"""Creates a Gaussian pyramid of a given image.

	This method uses reduce_image() at each level. Each image is
	stored in a list of length equal the number of levels.

	The first element in the list ([0]) should contain the input
	image. All other levels contain a reduced version of the previous
	level.

	All images in the pyramid should floating-point with values in

	Args:
		image (numpy.array): grayscale floating-point image, values
							 in [0.0, 1.0].
		levels (int): number of levels in the resulting pyramid.

	Returns:
		list: Gaussian pyramid, list of numpy.arrays.
	"""
	gaussian_pyramid_container = [np.copy(image)]
	for lvl in range(levels - 1):
		gaussian_pyramid_container.append(reduce_image(gaussian_pyramid_container[lvl]))
	return gaussian_pyramid_container


def create_combined_img(img_list):
	"""Stacks images from the input pyramid list side-by-side.

	Ordering should be large to small from left to right.

	See the problem set instructions for a reference on how the output
	should look like.

	Make sure you call normalize_and_scale() for each image in the
	pyramid when populating img_out.

	Args:
		img_list (list): list with pyramid images.

	Returns:
		numpy.array: output image with the pyramid images stacked
					 from left to right.
	"""
	combined_images = np.zeros(shape=(img_list[0].shape[0], np.sum([img.shape[1] for img in img_list])))
	prev = 0
	for img in img_list:
		combined_images[0:img.shape[0], prev:(prev + img.shape[1])] = img
		prev += img.shape[1]
	return combined_images * 255


def expand_image(image):
	"""Expands an image doubling its width and height.

	For simplicity and efficiency, implement a convolution-based
	method using the 5-tap separable filter.

	Follow the process shown in the lecture 6B-L3. Also refer to:
	-  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
	   as a Compact Image Code

	You can find the link in the problem set instructions.

	Args:
		image (numpy.array): grayscale floating-point image, values
							 in [0.0, 1.0].

	Returns:
		numpy.array: same type as 'image' with the doubled height and
					 width.
	"""
	
	# Larger image populated with zeros
	expanded_image = np.zeros(shape=(image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
	
	# Inserting empty pixels between pairs of pixels.
	expanded_image[::2, ::2] = image
	
	# this is the kernel specified in Piazza for Expand
	#  np.asarray([1, 4, 6, 4, 1]) / 8.0
	
	return cv2.filter2D(expanded_image, ddepth=-1,
	                    kernel=np.outer(np.asarray([1, 4, 6, 4, 1]) / 8.0,
	                                    np.asarray([1, 4, 6, 4, 1]) / 8.0),
	                    borderType=cv2.BORDER_REFLECT101)


def laplacian_pyramid(g_pyr):
	"""Creates a Laplacian pyramid from a given Gaussian pyramid.

	This method uses expand_image() at each level.

	Args:
		g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

	Returns:
		list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
	"""
	
	laplacian_pyramid_container = []
	for lvl in range(len(g_pyr)-1):
		gauss = g_pyr[lvl]
		expand_gauss = expand_image(g_pyr[lvl + 1])
		laplacian_pyramid_container.append(gauss - expand_gauss[:gauss.shape[0], :gauss.shape[1]])
		# Handle odd shaped images
	
	laplacian_pyramid_container.append(g_pyr[-1])
	return laplacian_pyramid_container


def warp(image, U, V, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT101):
	"""Warps image using X and Y displacements (U and V).

	This function uses cv2.remap. The autograder will use cubic
	interpolation and the BORDER_REFLECT101 border mode. You may
	change this to work with the problem set images.

	See the cv2.remap documentation to read more about border and
	interpolation methods.

	Args:
		image (numpy.array): grayscale floating-point image, values
							 in [0.0, 1.0].
		U (numpy.array): displacement (in pixels) along X-axis.
		V (numpy.array): displacement (in pixels) along Y-axis.
		interpolation (Inter): interpolation method used in cv2.remap.
		border_mode (BorderType): pixel extrapolation method used in
								  cv2.remap.

	Returns:
		numpy.array: warped image, such that
					 warped[y, x] = image[y + V[y, x], x + U[y, x]]
	"""
	# Getting the images height and width
	height, width = image.shape
	
	# Defining a meshgrid using the height and width
	mesh_x, mesh_y = np.meshgrid(np.arange(width), np.arange(height))
	
	return cv2.remap(image, (mesh_x + U).astype(np.float32), (mesh_y + V).astype(np.float32),
	                 interpolation=interpolation, borderMode=border_mode)


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
	"""Computes the optic flow using Hierarchical Lucas-Kanade.

	This method should use reduce_image(), expand_image(), warp(),
	and optic_flow_lk().

	Args:
		img_a (numpy.array): grayscale floating-point image, values in
							 [0.0, 1.0].
		img_b (numpy.array): grayscale floating-point image, values in
							 [0.0, 1.0].
		levels (int): Number of levels.
		k_size (int): parameter to be passed to optic_flow_lk.
		k_type (str): parameter to be passed to optic_flow_lk.
		sigma (float): parameter to be passed to optic_flow_lk.
		interpolation (Inter): parameter to be passed to warp.
		border_mode (BorderType): parameter to be passed to warp.

	Returns:
		tuple: 2-element tuple containing:
			U (numpy.array): raw displacement (in pixels) along X-axis,
							 same size as the input images,
							 floating-point type.
			V (numpy.array): raw displacement (in pixels) along Y-axis,
							 same size and type as U.
	"""
	gaussian_pyramid_a = gaussian_pyramid(image=img_a, levels=levels)
	gaussian_pyramid_b = gaussian_pyramid(image=img_b, levels=levels)

	U = np.zeros(shape=(gaussian_pyramid_a[-1].shape[0] // 2 + 1,
	                    gaussian_pyramid_a[-1].shape[1] // 2 + 1), dtype=np.float64)
	V = np.zeros(shape=(gaussian_pyramid_a[-1].shape[0] // 2 + 1,
	                    gaussian_pyramid_a[-1].shape[1] // 2 + 1), dtype=np.float64)

	for lvl in range(levels - 1, -1, -1):
		U = (expand_image(U) * 2)[:gaussian_pyramid_a[lvl].shape[0], :gaussian_pyramid_a[lvl].shape[1]]
		V = (expand_image(V) * 2)[:gaussian_pyramid_a[lvl].shape[0], :gaussian_pyramid_a[lvl].shape[1]]
		warped_b = warp(gaussian_pyramid_b[lvl], U, V, interpolation=interpolation, border_mode=border_mode)
		temp_u, temp_v = optic_flow_lk(img_a=gaussian_pyramid_a[lvl], img_b=warped_b, k_size=k_size,
		                               k_type=k_type, sigma=sigma)
		U += temp_u
		V += temp_v
	return U, V
