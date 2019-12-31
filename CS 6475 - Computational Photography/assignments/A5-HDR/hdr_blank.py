""" Building an HDR Image
This file has a number of functions that you need to complete. Please write
the appropriate code, following the instructions on which functions you may
or may not use.
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
Notation
--------
The following symbols are used in "Recovering High Dynamic Range Radiance
Maps from Photographs", by Debevec & Malik (available under resources on
T-Square), and are used extensively throughout the instructions:
    Z    : pixel intensity value; may be subscripted to indicate position
           in an array, e.g., Zij is the pixel intensity in row i column j
           of a 2D array
    Zmax : maximum pixel intensity (255 for uint8)
    Zmin : minimum pixel intensity (0 for uint8)
    W    : weight of an intensity Z; may be subscripted to indicate
           position in an array, e.g., Wk is the weight of Zk, the
           intensity at position k of a 1D array
    g    : response curve mapping pixel values Z to sensor response
    t    : frame exposure time; may be subscripted to indicate position in
           an array, e.g., ln(tj) is the log of exposure time of frame j
    E    : radiance value of a pixel; may be subscripted to indicate position
           in an array, e.g., ln(Ei) is the log radiance of pixel i
"""
import numpy as np
import scipy as sp
import cv2
import random


def linearWeight(pixel_value):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.
    The weighting function should be a piecewise linear function:
                           /  z - Zmin,      z <= (Zmin + Zmax)/2
        linearWeight(z) = |
                           \  Zmax - z,      z > (Zmin + Zmax)/2
    Where z is a pixel intensity value, Zmax=255 (largest uint8 intensity),
    and Zmin=0 (smallest uint8 intensity).
    Parameters
    ----------
    pixel_value : np.uint8
        A pixel intensity value from 0 to 255
    Returns
    -------
    weight : np.float64
        The weight corresponding to the input pixel intensity
    See Also
    --------
    "Recovering High Dynamic Range Radiance Maps from Photographs",
        Debevec & Malik (available under resources on T-Square)
    """
    z_min, z_max = 0., 255.
    # WRITE YOUR CODE HERE.
    raise NotImplementedError


def sampleIntensities(images):
    """ Randomly sample pixel intensities from the exposure stack.
    The returned `intensity_values` array has one row for every possible
    pixel value, and one column for each image in the exposure stack. The
    values in the array are filled according to the instructions below.
    Candidate locations are drawn from the middle image of the stack because
    it is expected to be the least likely image to contain over- or
    under-exposed pixels.
    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing a stack of single-channel (i.e., grayscale)
        layers of an HDR exposure stack
    Returns
    -------
    intensity_values : numpy.array, dtype=np.uint8
        An array containing a uniformly sampled intensity value from each
        exposure layer (shape = num_intensities x num_images)
    Notes
    -----
        (1) Remember that array coordinates (row, column) are in the opposite
            order as Cartesian coordinates (x, y).
    """
    # There are 256 intensity values to sample for uint8 images in the
    # exposure stack - one for each value [0...255], inclusive
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]  # using integer division is arbitrary

    # 1. Collect intensity samples from the image stack -- let Im be the
    # middle image in the exposure stack. Then, for each possible pixel
    # intensity level Zmin <= Zi <= Zmax:
    #
    #   Find the locations of all candidate pixels (pixels in Im with value Zi)
    #
    #       If there are no pixels in Im with value Zi, do nothing
    #
    #       Otherwise, randomly select a location (x, y) from the candidate
    #       locations and set intensity_values[i, j] to Zj, the intensity
    #       of image Ij from the image stack at location (x, y)
    #
    # WRITE YOUR CODE HERE.
    raise NotImplementedError


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    """ Find the camera response curve for a single color channel
    The constraints are described in detail in section 2.1 of "Recovering
    High Dynamic Range Radiance Maps from Photographs" by Debevec & Malik
    (available in the course resources material on T-Square) and see the
    description of the constraint matrix on google drive:
    https://drive.google.com/file/d/0B-R79qEgiV9dVERQQ1Z4R1NDOUE/view
    The "example.jpg" file further illustrates the correct structure of the
    constraint matrix. The example was generated for 3 images with 16 colors
    (you need to handle N images with 256 colors). The illustration just shows
    which pixels should be set by this function; it has a value of one in each
    location that was touched by this function. Your code needs to set the
    appropriate value in each location of the constraint matrix (some entries
    may have a value of 1, but that is not the correct value for all cells).
    You will first fill in mat_A and mat_b with coefficients corresponding to
    an overdetermined system of constraint equations, then solve for the
    response curve by finding the least-squares solution (i.e., solve for x
    in the linear system Ax=b).
        *************************************************************
            NOTE: Use the weighting_function() parameter to get
              the weight, do NOT directly call linearWeight()
        *************************************************************
    Parameters
    ----------
    intensity_samples : numpy.ndarray
        Stack of single channel input values (num_samples x num_images)
    log_exposures : numpy.ndarray
        Log exposure times (size == num_images)
    smoothing_lambda : float
        A constant value used to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.
    weighting_function : callable
        Function that computes a weight from a pixel intensity
    Returns
    -------
    numpy.ndarray, dtype=np.float64
        Return a vector g(z) where the element at index i is the log exposure
        of a pixel with intensity value z = i (e.g., g[0] is the log exposure
        of z=0, g[1] is the log exposure of z=1, etc.)
    """
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [Zmax - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints (the first NxP rows in the array).
    # For each of the k values in the range 0 <= k < intensity_samples.size
    # and the intensities Zij at (i, j) in the intensity_samples array:
    #
    #    Let Wij be the weight of Zij
    #
    #     i. Set mat_A at row k in column Zij to Wij
    #
    #    ii. Set mat_A at row k in column num_samples + i to -Wij
    #
    #   iii. Set mat_b at row k to Wij * log_exposure[j]
    #
    # WRITE YOUR CODE HERE


    # 2. Add smoothing constraints (the N-2 rows after the data constraints).
    # Beginning in the first row after the last data constraint, loop over each
    # value Zk in the range Zmin+1 <= Zk <= Zmax-1:
    #
    #   Let Wk be the weight of Zk
    #
    #     i. Set mat_A in the current row at column Zk - 1 to
    #        Wk * smoothing_lambda
    #
    #    ii. Set mat_A in the current row at column Zk to
    #        -2 * Wk * smoothing_lambda
    #
    #   iii. Set mat_A in the current row at column Zk + 1 to
    #        Wk * smoothing_lambda
    #
    #        Move to the next row
    #
    # WRITE YOUR CODE HERE


    # 3. Add color curve centering constraint (the last row of mat_A):
    #
    #     i. Set the value of mat_A in the last row and
    #        column (Zmax - Zmin) // 2 to the constant 1.
    #
    # WRITE YOUR CODE HERE


    # 4. Solve the system Ax=b. Recall from linear algebra that the solution
    # to a linear system can be obtained:
    #
    #   Ax = b
    #   A^-1 * A * x = b
    #   x = A^-1 * b
    #
    #   NOTE: The * operator here is the dot product, but the numpy *
    #         operator performs an element-wise multiplication
    #         (so don't use it -- use np.dot instead).
    #
    #     i. Get the Moore-Penrose psuedo-inverse of mat_A (Numpy has a
    #        function to do this)
    #
    #    ii. Multiply inv_A with mat_b (remember, use dot not *) to get x.
    #        If done correctly, x.shape should be 512 x 1
    #
    # WRITE YOUR CODE HERE
    raise NotImplementedError


    # Assuming that you set up your equation so that the first elements of
    # x correspond to g(z); otherwise you can change this to match your
    # constraints
    g = x[0:intensity_range + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    """ Calculate a radiance map for each pixel from the response curve.
    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)
    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)
    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z
    weighting_function : callable
        Function that computes the weights
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    """
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    # 1. Construct the radiance map -- for each pixel i in the output (note
    #    that "i" is a (row, col) location in this case):
    #
    #     i. Get all Zij values -- the intensities of pixel i from each
    #        image Ik in the input stack
    #
    #    ii. Get all Wij values -- the weight of each Zij (use the weighting
    #        function parameter)
    #
    #   iii. Calculate SumW - the sum of all Wij values for pixel i
    #
    #    iv. If SumW > 0, set pixel i in the output equal to the weighted
    #        average radiance (i.e., sum(Wij * (g(Zij) - ln(tj))) / SumW),
    #        otherwise set it to the log radiance from the middle image in
    #        the exposure stack (i.e., calculate the right hand side of
    #        Eq. 5: ln(Ei) = g(Zij) - ln(tj) from the source paper for
    #        just the middle image, rather than the average of the stack)
    #
    # WRITE YOUR CODE HERE
    raise NotImplementedError