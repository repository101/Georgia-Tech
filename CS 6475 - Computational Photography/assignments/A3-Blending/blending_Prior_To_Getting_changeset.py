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

TESTING = False
REDUCE = False
EXPAND = False
GAUSS = False
LAPL = False
BLEND = False
COLLAPSE = False


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
    if TESTING:
        the_seed = np.random.randint(0, 30)
    try:
        original_rows, original_cols = image.shape
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        convolved_image = cv2.filter2D(image, kernel=kernel, ddepth=-1, borderType=cv2.BORDER_REFLECT101)
        if convolved_image.dtype != np.float64:
            convolved_image = np.asarray(convolved_image, dtype=np.float64)
        reduced_rows = np.ceil(original_rows / 2)
        reduced_cols = np.ceil(original_cols / 2)
        reduced_image = convolved_image[::2, ::2]
        if reduced_image.dtype != np.float64:
            reduced_image = np.asarray(reduced_image, dtype=np.float64)
        result_rows, result_cols = reduced_image.shape
        if (result_rows != reduced_rows) or (result_cols != reduced_cols):
            reduced_image = reduced_image[:int(reduced_rows), :int(reduced_cols)]
        result_rows, result_cols = reduced_image.shape
        
        if TESTING and REDUCE:
            cv2.imwrite("Reduce_Layer_{}_Original_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, image.shape[0], image.shape[1],
                               np.random.randint(0, 20)), image)
            cv2.imwrite("Reduce_Layer_{}_Reduced_Image_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, reduced_image.shape[0], reduced_image.shape[1],
                               np.random.randint(0, 20)), reduced_image)
            cv2.imwrite("Reduce_Layer_{}_Convolved_Imag_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, convolved_image.shape[0], convolved_image.shape[1],
                               np.random.randint(0, 20)), convolved_image)

        if reduced_image.dtype != np.float:
            raise TypeError
        if reduced_rows != result_rows:
            raise ValueError
        if reduced_cols != result_cols:
            raise ValueError
        return reduced_image

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
    if TESTING:
        the_seed = np.random.randint(0, 30)
    try:
        original_rows, original_cols = image.shape
        expanded_rows = original_rows * 2
        expanded_cols = original_cols * 2
        expanded_layer = np.zeros(shape=(expanded_rows, expanded_cols), dtype=np.float64)
        expanded_layer[:expanded_rows:2, :expanded_cols:2] = image
        expanded_before_convolve = expanded_layer.copy()

        # Create the Expanded layer by convolving over the layer padded with 0's
        expanded_layer = np.asarray(
            cv2.filter2D(expanded_layer, kernel=kernel, ddepth=-1, borderType=cv2.BORDER_REFLECT101),
            dtype=np.float64)

        # Multiple Expanded Layer by 4
        expanded_layer_before_multiple = expanded_layer.copy()
        expanded_layer = expanded_layer * 4
        result_rows, result_cols = expanded_layer.shape
        
        if TESTING and EXPAND:
            cv2.imwrite("Expand_Layer_{}_Original_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, image.shape[0], image.shape[1],
                               np.random.randint(0, 20)), image)
            cv2.imwrite("Expand_Layer_{}_Expanded_Image_Shape_Before_Convolve_{}_{}_rand_{}.jpg".
                        format(the_seed, expanded_before_convolve.shape[0], expanded_before_convolve.shape[1],
                               np.random.randint(0, 20)), expanded_before_convolve)
            cv2.imwrite("Expand_Layer_{}_Convolved_Image_After_Multiply_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, expanded_layer.shape[0], expanded_layer.shape[1],
                               np.random.randint(0, 20)), expanded_layer)
            cv2.imwrite("Expand_Layer_{}_Convolved_Image_Before_Multiply_Shape_{}_{}_rand_{}.jpg".
                        format(the_seed, expanded_layer_before_multiple.shape[0],
                               expanded_layer_before_multiple.shape[1], np.random.randint(0, 20)),
                        expanded_layer_before_multiple)
        if expanded_layer.dtype != np.float:
            raise TypeError
        if result_rows != expanded_rows:
            raise ValueError
        if result_cols != expanded_cols:
            raise ValueError
        return expanded_layer
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
        if type(image) is not np.ndarray:
            image = np.asarray(image, dtype=np.float)
        if image.dtype != np.float:
            image = image.astype(dtype=np.float)
        gaussPyramidResults = [image]
        for gauss_level in range(levels):
            gaussPyramidResults.append(reduce_layer(gaussPyramidResults[gauss_level]).astype(dtype=np.float))
        
        if TESTING and GAUSS:
            cv2.imwrite("Gauss_Pyramid_Original_Image_{}_{}_rand_{}.jpg".
                        format(image.shape[0], image.shape[1], np.random.randint(0, 20)), image)
            for g_level in range(len(gaussPyramidResults)):
                cv2.imwrite("Gauss_Pyramid_Layer_{}_Image_{}_{}_rand_{}.jpg".
                            format(g_level, gaussPyramidResults[g_level].shape[0],
                                   gaussPyramidResults[g_level].shape[1], np.random.randint(0, 20)),
                            gaussPyramidResults[g_level])
                
        if not isinstance(gaussPyramidResults, list):
            raise TypeError
        if gaussPyramidResults[0].dtype != np.float:
            raise TypeError
        if len(gaussPyramidResults) != (levels + 1):
            raise ValueError
        return gaussPyramidResults
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
        lapl_results_list = []
        for lapl_layer in range(len(gaussPyr)):
            gauss_layer = gaussPyr[lapl_layer]
            if lapl_layer < (len(gaussPyr)-1):
                next_gauss_layer = gaussPyr[lapl_layer + 1]
                next_gauss_layer_height, next_gauss_layer_width = gauss_layer.shape
                expand = expand_layer(next_gauss_layer).astype(dtype=np.float)
            if expand.shape != gauss_layer.shape:
                expand = expand[:next_gauss_layer_height, :next_gauss_layer_width]
            if lapl_layer < (len(gaussPyr)-1):
                lapl_result = (gauss_layer - expand)
            else:
                lapl_result = gauss_layer
            lapl_results_list.append(lapl_result)
            if TESTING and LAPL:
                try:
                    cv2.imwrite("Laplacian_Pyramid_Gauss_Layer_{}_Expanded_Image_{}_{}_rand_{}.jpg".
                                format(lapl_layer, expand.shape[0], expand.shape[1],
                                       np.random.randint(0, 20)), expand)
                    cv2.imwrite("Laplacian_Pyramid_Gauss_Layer_{}_Result_Image_{}_{}_rand_{}.jpg".
                                format(lapl_layer, lapl_result.shape[1], lapl_result.shape[1],
                                       np.random.randint(0, 20)), lapl_result)
                    cv2.imwrite("Laplacian_Pyramid_Gauss_Layer_{}_Next_Gauss_Layer_Image_{}_{}_rand_{}.jpg".
                                format(lapl_layer, next_gauss_layer.shape[1], next_gauss_layer.shape[1],
                                       np.random.randint(0, 20)), next_gauss_layer)
                except Exception as TestingErr:
                    print("Error while testing. \n", TestingErr)
        if not isinstance(lapl_results_list, list):
            raise TypeError
        if lapl_results_list[0].dtype != np.float:
            raise TypeError
        return lapl_results_list
    except Exception as LaplPyramidException:
        print("Exception during the execution of laplPyramid function. \n", LaplPyramidException)


# noinspection PyUnboundLocalVariable
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
        for layer in range(len(laplPyrBlack)-1, -1, -1):
            # Get Black and White Images
            black_blend_image = laplPyrBlack[layer]
            white_blend_image = laplPyrWhite[layer]
            
            # Convert mask to int so it can be converted to bool
            mask = gaussPyrMask[layer].astype(int)
            white_mask = mask.astype(dtype=np.bool)
            black_mask = np.invert(white_mask)

            # Apply mask
            white_image_masked = white_blend_image * white_mask
            black_image_masked = black_blend_image * black_mask
            black_white_blend_result = black_image_masked + white_image_masked
            blend_result.append(black_white_blend_result)
            if TESTING and BLEND:
                try:
                    cv2.imwrite("Blend_Layer_{}_Black_Image_{}_{}_rand_{}.jpg".
                                format(layer, black_blend_image.shape[0], black_blend_image.shape[1],
                                       np.random.randint(0, 20)), black_blend_image)
                    cv2.imwrite("Blend_Layer_{}_Black_Image_Masked_{}_{}_rand_{}.jpg".
                                format(layer, black_image_masked.shape[0], black_image_masked.shape[1],
                                       np.random.randint(0, 20)), black_image_masked)
                    cv2.imwrite("Blend_Layer_{}_White_Image_{}_{}_rand_{}.jpg".
                                format(layer, white_blend_image.shape[0], white_blend_image.shape[1],
                                       np.random.randint(0, 20)), white_blend_image)
                    cv2.imwrite("Blend_Layer_{}_White_Image_Masked_{}_{}_rand_{}.jpg".
                                format(layer, white_image_masked.shape[0], white_image_masked.shape[1],
                                       np.random.randint(0, 20)), white_image_masked)
                    cv2.imwrite("Blend_Layer_{}_Black_And_White_Image_Masked_{}_{}_rand_{}.jpg".
                                format(layer, black_white_blend_result.shape[0], black_white_blend_result.shape[1],
                                       np.random.randint(0, 20)), black_white_blend_result)
                except Exception as TestingErr:
                    print("Exception while testing in Collapse. \n", TestingErr)
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
    try:
        for layer in range(len(pyramid)-1):
            base_image = pyramid[layer]
            if layer < (len(pyramid)-1):
                next_layer = pyramid[layer + 1]
            else:
                next_layer = pyramid[layer]
            next_layer_rows, next_layer_cols = next_layer.shape
            expanded_base = expand_layer(pyramid[layer])
            if expanded_base.shape != next_layer.shape:
                if layer == len(pyramid):
                    next_layer_rows = int(np.ceil(next_layer_rows * 2))
                    next_layer_cols = int(np.ceil(next_layer_cols * 2))
                    if (expanded_base.shape[0] != next_layer_rows) or (expanded_base.shape[1] != next_layer_cols):
                        expanded_base = expanded_base[:next_layer_rows, :next_layer_cols]
                else:
                    if expanded_base.shape != next_layer.shape:
                        expanded_base = expanded_base[:next_layer_rows, :next_layer_cols]
            if layer < len(pyramid):
                combined_layer = np.add(next_layer, expanded_base)
            else:
                combined_layer = expanded_base
            result_of_collapse = combined_layer
            if TESTING and COLLAPSE:
                try:
                    cv2.imwrite("Collapse_Layer_{}_Base_Image_{}_{}_rand_{}.jpg".
                                format(layer, base_image.shape[0], base_image.shape[1],
                                       np.random.randint(0, 20)), base_image)
                    cv2.imwrite("Collapse_Layer_{}_Expanded_Image_{}_{}_rand_{}.jpg".
                                format(layer, expanded_base.shape[0], expanded_base.shape[1],
                                       np.random.randint(0, 20)), expanded_base)
                    cv2.imwrite("Collapse_Layer_{}_Next_Layer_Image_{}_{}_rand_{}.jpg".
                                format(layer, next_layer.shape[0], next_layer.shape[1],
                                       np.random.randint(0, 20)), next_layer)
                    cv2.imwrite("Collapse_Layer_{}_Combined_Layers_Image_{}_{}_rand_{}.jpg".
                                format(layer, combined_layer.shape[0], combined_layer.shape[1],
                                       np.random.randint(0, 20)), combined_layer)
                except Exception as TestingErr:
                    print("Exception while testing in Collapse. \n", TestingErr)
        if not isinstance(result_of_collapse, (np.ndarray, np.array)):
            result_of_collapse = np.asarray(result_of_collapse, dtype=np.float)
        if result_of_collapse.shape != pyramid[len(pyramid)-1].shape:
            raise ValueError
        if result_of_collapse.dtype != np.float:
            raise TypeError
        return result_of_collapse
    except Exception as CollapseException:
        print("Exception when attempting to collapse the pyramid. \n", CollapseException)


if __name__ == "__main__":
    # black_files = ["black.jpg", "black_even_odd.jpg", "black_odd_even.jpg", "black_odd_odd.jpg"]
    # white_files = ["white.jpg", "white_even_odd.jpg", "white_odd_even.jpg", "white_odd_odd.jpg"]
    # mask_files = ["mask.jpg", "mask_even_odd.jpg", "mask_odd_even.jpg", "mask_odd_odd.jpg"]
    
    black_files = ["black.jpg"]
    white_files = ["white.jpg"]
    mask_files = ["mask.jpg"]
    
    # black_files = ["black.jpg", "black_odd_even.jpg", "black_even_odd.jpg", "black_odd_odd.jpg"]
    # white_files = ["white.jpg", "white_even_odd.jpg", "white_odd_odd.jpg", "white_odd_even.jpg"]
    # mask_files = ["mask.jpg", "mask_even_odd.jpg", "mask_odd_even.jpg", "mask_odd_odd.jpg"]
    for i in range(len(black_files)):
        black_image = cv2.imread(black_files[i], cv2.IMREAD_COLOR)
        black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
        white_image = cv2.imread(white_files[i], cv2.IMREAD_COLOR)
        white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_files[i], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        black_image = black_image.astype(np.uint8)
        white_image = white_image.astype(np.uint8)
        mask = mask.astype(np.uint8)
        gaussPy_black = gaussPyramid(black_image, 4)
        lapPy_black = laplPyramid(gaussPy_black)
        white_image = white_image.astype(np.uint8)
        gaussPy_white = gaussPyramid(white_image, 4)
        lapPy_white = laplPyramid(gaussPy_white)
        gaussMask = gaussPyramid(mask, 4)
        blended = blend(lapPy_white, lapPy_black, gaussMask)
        gaussPyramidResults = collapse(blended)
        cv2.imwrite("result.jpg", gaussPyramidResults)
    
    print("Finished")