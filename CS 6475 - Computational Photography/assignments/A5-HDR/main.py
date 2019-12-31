"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2
import numpy as np

import os
import sys
import errno

from os import path

import hdr as hdr
import time

TESTING = False


# Change the source folder and exposure times to match your own
# input images. Note that the response curve is calculated from
# a random sampling of the pixels in the image, so there may be
# variation in the output even for the example exposure stack
SRC_FOLDER = "images/source/sample"
EXPOSURE_TIMES = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
                             1 / 60.0, 1 / 40.0, 1 / 15.0])

OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def computeHDR(images, log_exposure_times, smoothing_lambda=100.):
    try:
        """Computational pipeline to produce the HDR images according to the
        process in the Debevec paper.
    
        NOTE: This function is NOT scored as part of this assignment.  You may
              modify it as you see fit.
    
        The basic overview is to do the following for each channel:
    
        1. Sample pixel intensities from random locations through the image stack
           to determine the camera response curve
    
        2. Compute response curves for each color channel
    
        3. Build image radiance map from response curves
    
        4. Apply tone mapping to fit the high dynamic range values into a limited
           range for a specific print or display medium (NOTE: we don't do this
           part except to normalize - but you're free to experiment.)
    
        Parameters
        ----------
        images : list<numpy.ndarray>
            A list containing an exposure stack of images
    
        log_exposure_times : numpy.ndarray
            The log exposure times for each image in the exposure stack
    
        smoothing_lambda : np.int (Optional)
            A constant value to correct for scale differences between
            data and smoothing terms in the constraint matrix -- source
            paper suggests a value of 100.
    
        Returns
        -------
        numpy.ndarray
            The resulting HDR with intensities scaled to fit uint8 range
        """
    
        images = [np.atleast_3d(i) for i in images]
    
        num_channels = images[0].shape[2]
    
        hdr_image = np.zeros(images[0].shape, dtype=np.float64)
        n = 0
        for channel in range(num_channels):
    
            # Collect the current layer of each input image from
            # the exposure stack
            layer_stack = [img[:, :, channel] for img in images]
    
            # Sample image intensities
            intensity_time_start = time.time()
            intensity_samples = hdr.sampleIntensities(layer_stack)
            intensity_time_end = time.time()
            print("Sample Intensity Time: {:.4f}sec\n".format(intensity_time_end-intensity_time_start))
    
            # Compute Response Curve
            response_time_start = time.time()
            response_curve = hdr.computeResponseCurve(intensity_samples,
                                                      log_exposure_times,
                                                      smoothing_lambda,
                                                      hdr.linearWeight)
            response_time_end = time.time()
            print("Response Curve Time: {:.4f}sec\n".format(response_time_end-response_time_start))
    
            # Build radiance map
            radiance_time_start = time.time()
            img_rad_map = hdr.computeRadianceMap(layer_stack,
                                                 log_exposure_times,
                                                 response_curve,
                                                 hdr.linearWeight)
            radiance_time_end = time.time()
            print("Radiance Map Time: {:.4f}sec\n".format(radiance_time_end - radiance_time_start))
    
            # We don't do tone mapping, but here is where it would happen. Some
            # methods work on each layer, others work on all the layers at once;
            # feel free to experiment.  If you implement tone mapping then the
            # tone mapping function MUST appear in your report to receive
            # credit.
            out = np.zeros(shape=img_rad_map.shape, dtype=img_rad_map.dtype)
            cv2.normalize(img_rad_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            hdr_image[..., channel] = out
            n += 1
        return hdr_image
    except Exception as err:
        print("ERRR: ", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def main(image_files, output_folder, exposure_times, resize=False):
    """Generate an HDR from the images in the source folder """
    try:
        total_time_start = time.time()
        # Print the information associated with each image -- use this
        # to verify that the correct exposure time is associated with each
        # image, or else you will get very poor results

        print("{:^30} {:>15}".format("Filename", "Exposure Time"))
        print("\n".join(["{:>30} {:^15.4f}".format(*v)
                         for v in zip(image_files, exposure_times)]))

        img_stack = [cv2.imread(name) for name in image_files
                     if path.splitext(name)[-1][1:].lower() in EXTENSIONS]
        
        if TESTING:
            print("Starting Testing Routine")
            testing_routine_start = time.time()
            test_hdr_image = cv2.imread("HDR_Image_Source.jpg")
            hdr.ReportingRoutine(img_stack, test_hdr_image)
            testing_routine_end = time.time()
            print("Total Run Time For Testing Routine: {:.4f}sec\n".format(testing_routine_end - testing_routine_start))

        if any([im is None for im in img_stack]):
            raise RuntimeError("One or more input files failed to load.")

        # Subsampling the images can reduce runtime for large files
        if resize:
            img_stack = [img[::4, ::4] for img in img_stack]

        log_exposure_times = np.log(exposure_times)
        # for i in range(len(img_stack)):
        #     hdr.GenerateHistogramsForReport(img_stack[i], count=i)
        hdr_image = computeHDR(img_stack, log_exposure_times)
        cv2.imwrite("HDR_Image_Source.jpg", hdr_image)

        cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)
        cv2.imwrite(path.join(output_folder, "exampleResult.jpg"), hdr_image)
        total_time_end = time.time()
        print("Total Run Time: {:.4f}sec\n".format(total_time_end - total_time_start))
        print("Done!")
    except Exception as err:
        print("ERRR: ", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == "__main__":
    """Generate an HDR image from the images in the SRC_FOLDER directory """

    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = next(src_contents)

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Processing '" + image_dir + "' folder...")

    image_files = sorted([os.path.join(dirpath, name) for name in fnames
                          if not name.startswith(".")])

    main(image_files, output_dir, EXPOSURE_TIMES, resize=False)
