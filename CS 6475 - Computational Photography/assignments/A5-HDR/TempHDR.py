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

TESTING = True


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
                                             hdr.linearWeight2)
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

        if any([im is None for im in img_stack]):
            raise RuntimeError("One or more input files failed to load.")

        # Subsampling the images can reduce runtime for large files
        if resize:
            img_stack = [img[::4, ::4] for img in img_stack]

        log_exposure_times = np.log(exposure_times)
        # for i in range(len(img_stack)):
        #     hdr.GenerateHistogramsForReport(img_stack[i], count=i)
        hdr_image = computeHDR(img_stack, log_exposure_times)
        if TESTING:
            print("GENERATING IMAGES FOR REPORT")
            HDR_Copy = hdr_image.copy()
            HDR_Copy_Histogram_Matched = hdr_image.copy()

            # Match Histograms
            b_source, g_source, r_source = cv2.split(img_stack[len(img_stack)//2]) # Source being the middle exposure image
            b_destination, g_destination, r_destination = cv2.split(HDR_Copy_Histogram_Matched)
            new_b = hdr.matchHistogramRoutine(b_source, b_destination)
            new_g = hdr.matchHistogramRoutine(g_source, g_destination)
            new_r = hdr.matchHistogramRoutine(r_source, r_destination)
            HDR_Copy_Histogram_Matched = cv2.merge((new_b, new_g, new_r))
            row_cord, col_cord = HDR_Copy_Histogram_Matched.shape[:2]
            loc_Y_1 = int(np.round(row_cord * 0.95))
            loc_X_1 = 20
            loc_Y_2 = int(np.round(row_cord * 0.99))
            loc_X_2 = 20
            line_space = 0
            font_size = 1

            # ToneMapBASIC (Gamma)
            for i in np.arange(0.1, 1.0, np.round(0.1, 1)):
                # ToneMapBASIC (Gamma)
                i = np.round(i, 1)
                img_title_1 = "Tone Map BASIC"
                img_title_2 = "Tone Map BASIC Matching Histogram"
                BASIC_tm = cv2.createTonemap(i)
                ToneMap_BASIC = HDR_Copy.copy().astype(np.float32)
                ToneMap_BASIC_Matched = HDR_Copy_Histogram_Matched.copy().astype(np.float32)
                temp = BASIC_tm.process(ToneMap_BASIC)
                temp_matched = BASIC_tm.process(ToneMap_BASIC_Matched)
                if temp.max() <= 2:
                    temp *= 255
                if temp_matched.max() <= 2:
                    temp_matched *= 255
                cv2.putText(temp, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp, "Gamma: {}".format(i), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp_matched, "Gamma: {}".format(i), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)

                cv2.imwrite("ToneMapBASIC/ToneMapBASIC_{}.jpg".format(i), temp)
                cv2.imwrite("ToneMapBASIC/ToneMapBASIC_Matched_{}.jpg".format(i), temp_matched)

                # ToneMapDrago (Bias, Saturation)
                for sat in np.arange(0.1, 1.0, np.round(0.1, 1)):
                    sat = np.round(sat, 1)
                    img_title_1 = "Tone Map Drago"
                    img_title_2 = "Tone Map Drago Matching Histogram"
                    Drago_tm = cv2.createTonemapDrago(gamma=i, saturation=sat)
                    ToneMap_Drago = HDR_Copy.copy().astype(np.float32)
                    ToneMap_Drago_Matched = HDR_Copy_Histogram_Matched.copy().astype(np.float32)
                    temp_drago = Drago_tm.process(ToneMap_Drago)
                    temp_drago_matched = Drago_tm.process(ToneMap_Drago_Matched)
                    if temp_drago.max() <= 2:
                        temp_drago *= 255
                    if temp_drago_matched.max() <= 2:
                        temp_drago_matched *= 255

                    cv2.putText(temp_drago, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_drago, "Gamma: {}  Saturation: {}".format(i, sat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_drago_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_drago_matched, "Gamma: {}  Saturation: {}".format(i, sat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.imwrite("ToneMapDrago/ToneMapDrago_Gamma{}_Saturation{}.jpg".format(i, sat), temp_drago)
                    cv2.imwrite("ToneMapDrago/ToneMapDrago_Matched_Gamma{}_Saturation{}.jpg".format(i, sat), temp_drago_matched)

                # ToneMapMantiuk (Saturation, Scale)
                for manSat in np.arange(0.1, 1.0, np.round(0.1, 1)):
                    manSat = np.round(manSat, 1)
                    img_title_1 = "Tone Map Mantiuk"
                    img_title_2 = "Tone Map Mantiuk Matching Histogram"
                    Mantiuk_tm = cv2.createTonemapMantiuk(gamma=i, saturation=manSat)
                    ToneMap_Mantiuk = HDR_Copy.copy().astype(np.float32)
                    ToneMap_Mantiuk_Matched = HDR_Copy_Histogram_Matched.copy().astype(np.float32)
                    temp_mantiuk = Mantiuk_tm.process(ToneMap_Mantiuk)
                    temp_mantiuk_matched = Mantiuk_tm.process(ToneMap_Mantiuk_Matched)
                    if temp_mantiuk.max() <= 2:
                        temp_mantiuk *= 255
                    if temp_mantiuk_matched.max() <= 2:
                        temp_mantiuk_matched *= 255
                    cv2.putText(temp_mantiuk, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_mantiuk, "Gamma: {}  Saturation: {}".format(i, manSat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_mantiuk_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_mantiuk_matched, "Gamma: {}  Saturation: {}".format(i, manSat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.imwrite("ToneMapMantiuk/ToneMapMantiuk_Gamma{}_Saturation{}.jpg".format(i, manSat), temp_mantiuk)
                    cv2.imwrite("ToneMapMantiuk/ToneMapMantiuk_Matched_Gamma{}_Saturation{}.jpg".format(i, manSat), temp_mantiuk_matched)

                # ToneMapReinhard (ColorAdaptation, Intensity, LightAdaptation)
                for adapt in np.arange(0.1, 1.0, np.round(0.1, 1)):
                    adapt = np.round(adapt, 1)
                    img_title_1 = "Tone Map Reinhard "
                    img_title_2 = "Tone Map Reinhard Matching Histogram"
                    Reinhard_tm = cv2.createTonemapReinhard(gamma=i, color_adapt=adapt)
                    ToneMap_Reinhard = HDR_Copy.copy().astype(np.float32)
                    ToneMap_Reinhard_Matched = HDR_Copy_Histogram_Matched.copy().astype(np.float32)
                    temp_reinhard = Reinhard_tm.process(ToneMap_Reinhard)
                    temp_reinhard_matched = Reinhard_tm.process(ToneMap_Reinhard_Matched)
                    if temp_reinhard.max() <= 2:
                        temp_reinhard *= 255
                    if temp_reinhard_matched.max() <= 2:
                        temp_reinhard_matched *= 255
                    cv2.putText(temp_reinhard, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX,
                                font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_reinhard, "Gamma: {}  Color Adaptation: {}".format(i, adapt), (loc_X_2, loc_Y_2),
                                cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_reinhard_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1),
                                cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_reinhard_matched, "Gamma: {}  Color Adaptation: {}".format(i, adapt), (loc_X_2, loc_Y_2),
                                cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.imwrite("ToneMapReinhard/ToneMapReinhard_Gamma{}_ColorAdaption{}.jpg".format(i, adapt), temp_reinhard)
                    cv2.imwrite("ToneMapReinhard/ToneMapReinhard_Matched_Gamma{}_ColorAdaption{}.jpg".format(i, adapt), temp_reinhard_matched)

                # ToneMapDurand (Contrast, Saturation, SigmaColor, SigmaSpace)
                for durSat in np.arange(0.1, 1.0, np.round(0.1, 1)):
                    durSat = np.round(durSat, 1)
                    img_title_1 = "Tone Map Durand"
                    img_title_2 = "Tone Map Durand Matching Histogram"
                    Durand_tm = cv2.createTonemapDurand(gamma=i, saturation=durSat)
                    ToneMap_Durand = HDR_Copy.copy().astype(np.float32)
                    ToneMap_Durand_Matched = HDR_Copy_Histogram_Matched.copy().astype(np.float32)
                    temp_durand = Durand_tm.process(ToneMap_Durand)
                    temp_durand_matched = Durand_tm.process(ToneMap_Durand_Matched)
                    if temp_durand.max() <= 2:
                        temp_durand *= 255
                    if temp_durand_matched.max() <= 2:
                        temp_durand_matched *= 255

                    cv2.putText(temp_durand, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_durand, "Gamma: {}  Saturation: {}".format(i, durSat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size,
                                (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_durand_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.putText(temp_durand_matched, "Gamma: {}  Saturation: {}".format(i, durSat), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX,
                                font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                    cv2.imwrite("ToneMapDurand/ToneMapDurand_Gamma{}_Saturation{}.jpg".format(i, durSat), temp_durand)
                    cv2.imwrite("ToneMapDurand/ToneMapDurand_Matched_Gamma{}_Saturation{}.jpg".format(i, durSat), temp_durand_matched)

                # ToneMap Alternate Routine
                img_title_1 = "Tone Map Alternate"
                img_title_2 = "Tone Map Alternate Matching Histogram"
                ToneMap_Alternate = HDR_Copy.copy()
                ToneMap_Alternate_Matched = HDR_Copy_Histogram_Matched.copy()
                temp_Alternate = hdr.AlternateHistogramRoutine(ToneMap_Alternate, i)
                temp_Alternate_matched = hdr.AlternateHistogramRoutine(ToneMap_Alternate_Matched, i)
                if temp_Alternate.max() <= 2:
                    temp_Alternate *= 255
                if temp_Alternate_matched.max() <= 2:
                    temp_Alternate_matched *= 255
                cv2.putText(temp_Alternate, "{}".format(img_title_1), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp_Alternate, "Fraction: {}".format(i), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp_Alternate_matched, "{}".format(img_title_2), (loc_X_1, loc_Y_1), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.putText(temp_Alternate_matched, "Fraction: {}".format(i), (loc_X_2, loc_Y_2), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), line_space, cv2.LINE_AA)
                cv2.imwrite("ToneMapAlternate/ToneMapAlternate_{}.jpg".format(i), temp_Alternate)
                cv2.imwrite("ToneMapAlternate/ToneMapAlternate_Matched_{}.jpg".format(i), temp_Alternate_matched)


            # hdr.GenerateHistogramsForReport(new_HDR, count=200, filename="Histogram Matched HDR")

        cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)
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
    test = cv2.imread("test_image.jpg")
    row, col = test.shape[:2]
    cv2.putText(test, "{}".format("Test Image Format Stuff"), (20, int(np.round(row * 0.95))), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 0, cv2.LINE_AA)
    cv2.putText(test, "Gamma: {}".format("0.1"), (20, int(np.round(row * 0.99))), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 0, cv2.LINE_AA)
    cv2.imwrite("ResultingTestImage.jpg", test)

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
