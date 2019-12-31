""" Pyramid Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available on T-square under references:

(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983

(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983

Notes
-----
    You may not use cv2.pyrUp or cv2.pyrDown anywhere in this assignment.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.

    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2

#REMOVE
import matplotlib.pyplot as plt


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter (i.e., a
    square "5-tap" filter.)

    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel and then reduce its
    width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                          fefghg
        abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   fefghg   -------->  VUTS   -------->   RP
        ijkl    BORDER    jijklk     keep     RQPO               JH
        mnop   REFLECT    nmnopo     valid    NMLK
        qrst              rqrsts              JIHG
                          nmnopo

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """
    
    # WRITE YOUR CODE HERE.
    try:
        original_height, original_width = image.shape
        convolved_image = np.asarray(
            cv2.filter2D(image, kernel=kernel, ddepth=-1, borderType=cv2.BORDER_REFLECT101), dtype=np.float64)
        if convolved_image.dtype != np.float64:
            try:
                convolved_image = np.asarray(convolved_image, dtype=np.float64)
            except Exception as ToFloatException:
                print("Exception when attempting to convert array to dtype np.float64", ToFloatException)
        try:
            reduced_height = np.ceil(original_height / 2)
            reduced_width = np.ceil(original_width / 2)
            reduced_image = convolved_image[0:original_height:2, 0:original_width:2]
            if reduced_image.dtype != np.float64:
                try:
                    reduced_image = np.asarray(reduced_image, dtype=np.float64)
                except Exception as ConvertDTypeForReturn:
                    print("Exception when attempting to convert the reduced image arrays dtype"
                          " to np.float64. \n", ConvertDTypeForReturn)
            return reduced_image
        except Exception as scaleArrayError:
            print("Exception when attempting to scale down the array by a factor or 2 in the reduce_layer function. \n",
                  scaleArrayError)
    except Exception as reduce_layer_exception:
        print("Error during reduce_layer: \n", reduce_layer_exception)


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel.

    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after). For
    grading purposes, it is important that you use a reflected border (i.e.,
    padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid region
    (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                                          000000
             Upsample   A0B0     Pad      0A0B0B   Convolve   zyxw
        AB   ------->   0000   ------->   000000   ------->   vuts
        CD              C0D0    BORDER    0C0D0D     keep     rqpo
        EF              0000   REFLECT    000000    valid     nmlk
                        E0F0              0E0F0F              jihg
                        0000              000000              fedc
                                          0E0F0F

                NOTE: Remember to multiply the output by 4.

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """
    
    # WRITE YOUR CODE HERE.
    try:
        original_height, original_width = image.shape
        expanded_height = original_height * 2
        expanded_width = original_width * 2
        try:
            expanded_layer = np.zeros(shape=(expanded_height, expanded_width), dtype=np.float64)
            expanded_layer[0:expanded_height:2, 0:expanded_width:2] = image
            try:
                # Create the Expanded layer by convolving over the layer padded with 0's
                expanded_layer = np.asarray(
                    cv2.filter2D(expanded_layer, kernel=kernel, ddepth=-1, borderType=cv2.BORDER_REFLECT101),
                    dtype=np.float64)
                
                # Multiple Expanded Layer by 4
                expanded_layer = expanded_layer * 4
            except Exception as ConvolutionException:
                print("Exception when attempting to convolve over the expanded layer. \n", ConvolutionException)
            return expanded_layer
        except Exception as ExpandedArrayCreationException:
            print("Exception when attempting to create expanded array. \n", ExpandedArrayCreationException)
    except Exception as ExpandLayerException:
        print("Exception inside of expand_layer function. \n", ExpandLayerException)


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.

    You must use your reduce_layer() function to generate the pyramid.

    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """
    
    # WRITE YOUR CODE HERE.
    try:
        try:
            if type(image) is not np.ndarray:
                try:
                    image = np.asarray(image, dtype=np.float)
                except Exception as err:
                    print("Error when attempting to convert the input image for gauss pyramid to np array. \n", err)
            if image.dtype != np.float:
                image = image.astype(dtype=np.float)
        except Exception as ConvertException:
            print("Exception when attempting to convert input array to correct dtype. \n", ConvertException)
        results = [image]
        for i in range(levels):
            if i == 0:
                results.append(reduce_layer(results[0]).astype(dtype=np.float))
            else:
                results.append(reduce_layer(results[i]).astype(dtype=np.float))
        return results
    except Exception as GaussPyramidException:
        print("Exception when attempting to execute Gauss Pyramid. \n", GaussPyramidException)


def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    You must use your expand_layer() function to generate the pyramid. The
    Gaussian Pyramid that is passed in is the output of your gaussPyramid
    function.

    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.

    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.

        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """
    
    # WRITE YOUR CODE HERE.
    try:
        lapl_results = []
        for i in range(len(gaussPyr)):
            gauss = gaussPyr[i]
            gauss_height, gauss_width = gauss.shape
            if i >= (len(gaussPyr) - 1):
                expand = expand_layer(gaussPyr[i]).astype(dtype=np.float)
            else:
                expand = expand_layer(gaussPyr[i + 1]).astype(dtype=np.float)
            try:
                if gauss.shape != expand.shape:
                    expand = cv2.resize(expand, (gauss_width, gauss_height))
            except Exception as ConvertLaplException:
                print("Exception when attempting to resize arrays in laplpyramid function.", ConvertLaplException)
                
            lapl_results.append(gauss - expand)
        return lapl_results
    except Exception as LaplPyramidException:
        print("Exception during the execution of laplPyramid function. \n", LaplPyramidException)


def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)

    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.

    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.

    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids

    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """
    
    # WRITE YOUR CODE HERE.
    try:
        blend_result = []
        for img in range(len(laplPyrBlack)):
            black_img = laplPyrBlack[img]
            white_img = laplPyrWhite[img]
            mask = gaussPyrMask[img].astype(int)
            white_mask = mask.astype(dtype=np.bool)
            black_mask = np.invert(white_mask)
            white_result = white_img * white_mask
            black_result = black_img * black_mask
            result = black_result + white_result
            blend_result.append(result)
        for i in range(len(blend_result)):
            cv2.imwrite("{}.jpg".format(i), blend_result[i])
        return blend_result
    except Exception as BlendException:
        print("Exception when running blend function. \n", BlendException)
        

def collapse(pyramid):
    """Collapse an input pyramid.

    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.

        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """
    
    # WRITE YOUR CODE HERE.
    raise NotImplementedError


if __name__ == "__main__":
    black_image = cv2.imread("black.jpg", cv2.IMREAD_COLOR)
    black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    
    white_image = cv2.imread("white.jpg", cv2.IMREAD_COLOR)
    white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.imread("mask.jpg", cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # temp = generatingKernel(0.4)
    # black_image_reduced = reduce_layer(black_image, temp)
    # black_image_expanded = expand_layer(black_image_reduced, temp)
    
    gaussPy_black = gaussPyramid(black_image, 4)
    lapPy_black = laplPyramid(gaussPy_black)

    gaussPy_white = gaussPyramid(white_image, 4)
    lapPy_white = laplPyramid(gaussPy_white)
    
    gaussMask = gaussPyramid(mask, 4)
    
    blended = blend(lapPy_white, lapPy_black, gaussMask)

    print("The End")
    