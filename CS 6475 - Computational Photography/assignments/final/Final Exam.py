import numpy as np
import pandas as pd
import cv2


def find_corners(img):
    """Find corners in an image using Harris corner detection method."""
    
    # Convert to grayscale, if necessary
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Compute Harris corner detector response (params: block size, Sobel aperture, Harris alpha)
    h_response = cv2.cornerHarris(img_gray, 4, 3, 0.04)
    
    # Display Harris response image
    h_min, h_max, _, _ = cv2.minMaxLoc(h_response)  # for thresholding, display scaling
    cv2.imshow("Harris response", np.uint8((h_response - h_min) * (255.0 / (h_max - h_min))))
    
    # Select corner pixels above threshold
    h_thresh = 0.01 * h_max
    _, h_selected = cv2.threshold(h_response, h_thresh, 1, cv2.THRESH_TOZERO)
    
    # Pick corner pixels that are local maxima
    nhood_size = 5  # neighborhood size for non-maximal suppression (odd)
    nhood_r = int(nhood_size / 2)  # neighborhood radius = size / 2
    corners = []  # list of corner locations as (x, y, response) tuples
    for y in range(h_selected.shape[0]):
        for x in range(h_selected.shape[1]):
            if h_selected.item(y, x):
                h_value = h_selected.item(y, x)  # response value at (x, y)
                nhood = h_selected[(y - nhood_r):(y + nhood_r + 1), (x - nhood_r):(x + nhood_r + 1)]
                if not nhood.size:
                    continue  # skip empty neighborhoods (which can happen at edges)
                local_max = np.amax(nhood)  # compute neighborhood maximum
                if h_value == local_max:
                    corners.append((x, y, h_value))  # add to list of corners
                    h_selected[(y - nhood_r):(y + nhood_r), (x - nhood_r):(x + nhood_r)] = 0  # clear
                    h_selected.itemset((y, x), h_value)  # retain maxima value to suppress others
    
    h_suppressed = np.uint8((h_selected - h_thresh) * (255.0 / (h_max - h_thresh)))
    cv2.imshow("Suppressed Harris response", h_suppressed)
    return corners


def test_corners():
    """Test find_corners() with sample input."""
    
    # Read image
    img = cv2.imread("octagon.png")
    cv2.imshow("Image", img)
    
    # Find corners
    corners = find_corners(img)
    print("\n".join("{} {}".format(corner[0], corner[1]) for corner in corners))
    
    # Display output image with corners highlighted
    img_out = img.copy()
    for (x, y, resp) in corners:
        cv2.circle(img_out, (x, y), 1, (0, 0, 255), -1)  # red dot
        cv2.circle(img_out, (x, y), 5, (0, 255, 0), 1)  # green circle
    cv2.imshow("Output", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_anaglyph():
    """Driver function called by Test Run."""
    img_left = cv2.imread("flowers-left.png", 0)  # grayscale
    img_right = cv2.imread("flowers-right.png", 0)  # grayscale
    cv2.imshow("Left image", img_left)
    cv2.imshow("Right image", img_right)

    img_ana = make_anaglyph(img_left, img_right)
    cv2.imshow("Anaglyph image", img_ana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def make_anaglyph(img_left, img_right):
    """Create a red/cyan anaglyph image using grayscale left and right images."""
    # TODO: Combine images in appropriate color channels and return
    return np.dstack([img_right, img_right, img_left])


def Calculate_R():
    Ix = np.asarray([[0, 1, 0],
                     [1, 0, 1],
                     [1, 0, 1]])
    Iy = np.asarray([[0, 1, 1],
                     [1, 0, 0],
                     [0, 1, 1]])
    kappa = 0.05
    M_Ixx = cv2.boxFilter(Ix * Ix, cv2.CV_32S, (3, 3), normalize=False)[1, 1]
    M_Iyy = cv2.boxFilter(Iy * Iy, cv2.CV_32S, (3, 3), normalize=False)[1, 1]
    M_Ixy = cv2.boxFilter(Ix * Iy, cv2.CV_32S, (3, 3), normalize=False)[1, 1]
    R = (M_Ixx * M_Iyy - M_Ixy * M_Ixy) - kappa * (M_Ixx + M_Iyy)**2
    print("Here is R: \n", R)
    return
    
    
def Calculate_Magnitude_Histogram():
    magnitudes = np.array([[8, 3],
                           [7, 5]])
    angles = np.array([[76, 101],
                       [347, 154]])
    nbins = 4
    # bins = np.int64((2 * nbins * angles / 360) % nbins)  # unsigned gradients
    bins = np.int64(nbins * angles / 360)  # signed gradients
    print("Here are the Bins: \n", bins)
    histogram = np.zeros((1, 1, nbins))
    for k in range(nbins):
        histogram[0, 0, k] = np.sum(magnitudes[bins == k])
    print("Here is the Histogram: \n", histogram)
    return


def Calculate_HOG():
    window_size = (128, 64)
    cell_size = (8, 8)
    nbins = 9
    hist_shape = (window_size[0] // cell_size[0], window_size[1] // cell_size[1], nbins)
    block_size = (3, 3)
    row_blocks = hist_shape[0] - block_size[0] + 1
    col_blocks = hist_shape[1] - block_size[1] + 1
    block_vector_size = block_size[0] * block_size[1] * nbins
    blocks = np.zeros((row_blocks, col_blocks, block_vector_size))
    block_size = blocks.size
    print("The block size: \n", block_size)
    return
    
    
if __name__ == "__main__":
    rows = 4
    Calculate_R()
    Calculate_Magnitude_Histogram()
    Calculate_HOG()
    test_corners()
    test_anaglyph()
    columns = 5
    array = np.zeros(shape=(rows, columns))
    filter_row, filter_col = 5, 5
    arr = np.zeros_like(array)
    averageFilter = cv2.blur(arr, (filter_row, filter_col))
    gaussianFilter = cv2.GaussianBlur(arr, (filter_row, filter_col), 0)
    print()
'''
    Elements of Computational Photography
        1. Illumination
        2. Optics
        3. Sensor
        4. Processing
        5. Display
        6. User
    
    
    Types of Panaoramas
        1. Planar/Rectilinear
        2. Rotational/Spherical/Cylindrical
        3. Path/Route/Multiview
        4. Vertical
    
    Bits to represent a pixel
        1. you need 8 bits --- 2^8 = 256 => 0-255
    
    Any image convolved with a box filter will have an -- [Average Output]
    Any image cross-correlated with a gaussian filter will have a -- [Blurred Output]
    An impulse image convolved with a box filter will have -- [Averaged Output]
    An impulse cross-correlated with a gaussian filter will have -- [Blurred Output]
    The result of Convolution and Cross-Correlation is the same when -- [The kernel used is symmetric in both X and Y]

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
'''
    