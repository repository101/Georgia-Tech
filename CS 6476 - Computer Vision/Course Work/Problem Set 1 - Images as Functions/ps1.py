import math
import numpy as np
import cv2

# # Implement the functions below.


def find_middle(image):
    return {"Y": image.shape[0] // 2, "X": image.shape[1] // 2}


def min_max_normalization(arr):
    if (arr.max() - arr.min()) == 0:
        return np.zeros(shape=arr.shape)
    return (arr - arr.min()) / (arr.max() - arr.min())


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    return temp_image[:, :, 2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    return temp_image[:, :, 1]


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    return temp_image[:, :, 0]


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    # Swap Green to Blue
    temp_image[:, :, 1] = image[:, :, 0]

    # Swap Blue to Green
    temp_image[:, :, 0] = image[:, :, 1]

    return temp_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """

    src_center = find_middle(src)
    dst_center = find_middle(dst)
    shape_center = {"Y": shape[0] // 2, "X": shape[1] // 2}
    shape_obj = src[src_center["Y"] - shape_center["Y"]:src_center["Y"] + shape_center["Y"],
                src_center["X"] - shape_center["X"]:src_center["X"] + shape_center["X"]]
    temp_img = np.copy(dst)
    temp_img[dst_center["Y"] - shape_center["Y"]: dst_center["Y"] + shape_center["Y"],
    dst_center["X"] - shape_center["X"]: dst_center["X"] + shape_center["X"]] = shape_obj
    return temp_img


def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src_center = find_middle(src)
    dst_center = find_middle(dst)
    temp_src_image = np.copy(src).astype(np.float)
    temp_dst_image = np.copy(dst).astype(np.float)

    # Old Way not working for gradscope
    # src_x, src_y = np.meshgrid(np.arange(0, src.shape[1]), np.arange(0, src.shape[0]))
    # dst_x, dst_y = np.meshgrid(np.arange(0, dst.shape[1]), np.arange(0, dst.shape[0]))
    # src_mask = (src_y - src_center["Y"]) ** 2 + (src_x - src_center["X"]) ** 2 < (radius ** 2) + 1
    # dst_mask = (dst_y - dst_center["Y"]) ** 2 + (dst_x - dst_center["X"]) ** 2 < (radius ** 2) + 1
    # temp_dst_image[dst_mask] = temp_src_image[src_mask]
    src_mask = np.zeros(shape=temp_src_image.shape, dtype=np.float)
    dst_mask = np.zeros(shape=temp_dst_image.shape, dtype=np.float)
    src_mask = cv2.circle(src_mask, (src_center["X"], src_center["Y"]), radius, 1, -1).astype(np.bool)
    dst_mask = cv2.circle(dst_mask, (dst_center["X"], dst_center["Y"]), radius, 1, -1).astype(np.bool)
    temp_dst_image[dst_mask] = temp_src_image[src_mask]
    return temp_dst_image


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image).astype(np.float)
    return np.min(temp_image), np.max(temp_image), np.mean(temp_image), np.std(temp_image)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.copy(image).astype(np.float)
    temp_image = (temp_image - np.mean(temp_image)) / np.std(temp_image)

    # Multiply by Scaling Factor
    temp_image *= scale

    return temp_image


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    test = np.zeros(shape=temp_image.shape)
    test[:, :test.shape[1]-shift] = temp_image[:, shift:]
    for i in range(shift):
        test[:, test.shape[1]-shift+i] = temp_image[:, -1]
    return test.astype(np.uint8)


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    temp_image_1 = np.copy(img1).astype(np.float)
    temp_image_2 = np.copy(img2).astype(np.float)
    result = center_and_normalize(temp_image_1 - temp_image_2, 1)
    return ((result - result.min()) / (result.max() - result.min())) * 255


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image).astype(np.float)
    # temp_image[:, :, channel] += np.random.normal(0, sigma, size=(temp_image[:,
    #                                                               :, channel].shape[0],
    #                                                               temp_image[:, :, channel].shape[1]))
    temp_image[:, :, channel] += (np.random.randn(temp_image[:, :, channel].shape[0],
                                                  temp_image[:, :, channel].shape[1]) * sigma)
    return temp_image


def build_hybrid_image(image1, image2, cutoff_frequency):
    """
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar

    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays.
    """
    image1 = image1.astype(np.float)
    image2 = image2.astype(np.float)
    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    low_frequencies = cv2.filter2D(image1, -1, filter)
    high_frequencies = (image2 - cv2.filter2D(image2, -1, filter))

    # Best Attempt Still Wrong
    # temp = (min_max_normalization((low_frequencies + high_frequencies)) * 255).astype(np.uint8)
    temp = min_max_normalization(center_and_normalize(low_frequencies + high_frequencies, 1)) * 255
    return temp.astype(np.uint8)


def vis_hybrid_image(hybrid_image):
    """
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays.
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
