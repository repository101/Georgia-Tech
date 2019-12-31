import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn
import sklearn

import argparse

from FinalHelper import *

RESIZE = True     # Used just for resizing images
TESTING = True      # Global used in testing
EXTRACTFRAMES = False       # Used to extract frames from videos, if needed
ALIGN = False       # Used to Align images before processing, used for testing only


def getCurrentWorkingDirectory():
    try:
        return os.getcwd()
    except Exception as GetCurrentWorkingDirectoryException:
        print("Exception occurred while attempting to execute function getCurrentWorkingDirectory. \n",
              GetCurrentWorkingDirectoryException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def alignImages(image_to_align, reference_image):
    # I DID NOT MAKE THIS
    # https: // www.learnopencv.com / image - alignment - feature - based - using - opencv - c - python /
    # was used for testing
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(10000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(image_to_align, keypoints1, reference_image, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width, channels = reference_image.shape
    im1Reg = cv2.warpPerspective(image_to_align, h, (width, height))
    
    return im1Reg, h


def getFramesFromVideo(pathToVideo, saveDirectory):
    try:
        # Source https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
        # I did not make this, I used it to get the frames from the video used in the paper
        # cv2.readOpticalFlow()
        video_capture = cv2.VideoCapture(pathToVideo)
        success, image = video_capture.read()
        count = 0
        while success:
            # save frame as JPEG file
            # 'frame{0:04d}.png'.format(idx)
            cv2.imwrite(saveDirectory + "/frame{0:04d}.png".format(count), image)
            success, image = video_capture.read()
            count += 1
        return
    except Exception as GetFramesFromVideoException:
        print("Exception occurred while executing GetFramesFromVideo. \n", GetFramesFromVideoException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def optical_flow(I1g, I2g, window_size, tau=1e-2):
    # https: // sandipanweb.wordpress.com / 2018 / 02 / 25 / implementing - lucas - kanade - optical - flow - algorithm - in -python /
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size / 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t,
                                                                                          boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            # b = ... # get b here
            # A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ...  # get velocity here
            u[i, j] = nu[0]
            v[i, j] = nu[1]
    
    return (u, v)


def getImageFileNames(image_dir):
    try:
        extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                      'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']
        
        search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
        image_files = sorted(sum(map(glob, search_paths), []))
        return image_files
    except Exception as GetImageFileNamesException:
        print("Exception occurred while attempting to get the image file names. \n", GetImageFileNamesException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def generateGif(files, dur=0.1, name=None, ATWORK=False):
    try:
        dir = "C:/Users/joshu/"
        if ATWORK:
            dir = "C:/Users/Josh.Adams/"
        with imageio.get_writer("{}OneDrive - Georgia Institute of "
                                "Technology/Georgia-Tech/CS 6475 - Computational "
                                "Photography/assignments/Final_Project/{}.gif".format(dir, name),
                                mode="I", duration=dur) as writer:
            for filename in files:
                image = imageio.imread(filename)
                writer.append_data(image)
        return
    except Exception as GenerateGifException:
        print("Exception occurred while executing GenerateGif. \n", GenerateGifException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def resizeImages(filePaths, percent):
    try:
        img_dir = Path(filePaths[0]).parents[0]
        output_directory_percent = os.path.join(str(img_dir), "Resized_{}%_Images".format(percent * 100))
        # Check if directories exist and create if they do not
        # Source https://stackoverflow.com/questions/31008598/python-check-if-a-directory-exists-then-create-it-if-necessary-and-save-graph-t?noredirect=1&lq=1
        
        if not os.path.exists(output_directory_percent):
            os.mkdir(output_directory_percent)
        for img in filePaths:
            current_file_name = img.split("\\")[-1]
            temp_out_1 = os.path.join(output_directory_percent, current_file_name)
            # Read Image
            temp_img = cv2.imread(img, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
            # Resize Images
            resized_img_1 = cv2.resize(temp_img, None, fx=percent, fy=percent)
            # Save Images
            cv2.imwrite(temp_out_1, resized_img_1)
        return output_directory_percent
    except Exception as ResizeImageException:
        print("Exception occurred while attempting to resize images. \n", ResizeImageException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def Setup(optionalPath, pct=None):
    try:
        # directory = "Frames/Resized_15.0%_Images"
        directory = "Frames/"
        if optionalPath.image_directory is None:
            imageDirectory = getCurrentWorkingDirectory()
        else:
            imageDirectory = optionalPath.image_directory
        image_file_paths = ""
        # This will get the working directory and get everything setup that is needed
        if EXTRACTFRAMES:
            pathToVideo = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/" \
                          "Georgia-Tech/CS 6475 - Computational Photography/" \
                          "assignments/Final_Project/Data/siggraph_input.avi"
            pathToSaveDirectory = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/" \
                                  "Georgia-Tech/CS 6475 - Computational Photography/" \
                                  "assignments/Final_Project/SigGraph_Frames"
            getFramesFromVideo(pathToVideo, pathToSaveDirectory)
        if TESTING and not RESIZE:
            imageDirectory = os.path.join(imageDirectory, directory)
            if not os.path.exists(imageDirectory):
                os.mkdir(imageDirectory)
            image_file_paths = getImageFileNames(imageDirectory)
            if len(image_file_paths) <= 1:
                sys.exit()
        if RESIZE:
            # Read Images
            image_file_paths = getImageFileNames(os.path.join(imageDirectory, directory))
            output = resizeImages(image_file_paths, percent=pct)
            image_file_paths = getImageFileNames(output)
            return image_file_paths
        elif not TESTING:
            image_file_paths = getImageFileNames(
                os.path.join(imageDirectory, "Frames//Resized_{}%_Images".format(str(pct))))

        return image_file_paths
    except Exception as SetupException:
        print("Exception occurred while setting up application. \n", SetupException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def main():
    try:
        parser = argparse.ArgumentParser(description='Final Project Application')
        parser.add_argument('-o', '--image_directory', help='Output file name',
                            type=str, default='', nargs='?', required=False)

        args = parser.parse_args()
        if RESIZE:
            image_paths = Setup(args, 0.15)
        else:
            image_paths = Setup(args)
        
        # Align Images
        # if ALIGN:
        #     reference_image = cv2.imread(image_paths[2], cv2.IMREAD_COLOR)
        #     for i in range(len(image_paths)):
        #         image_to_be_aligned = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        #         warped_image, homography = alignImages(image_to_align=image_to_be_aligned, reference_image=reference_image)
        #         cv2.imwrite("Frames/test_" + image_paths[i].split("\\")[-1], warped_image)

        App_Object = ImageObstructionClean(image_paths=image_paths, reference_image_index=2)
        print()
    except Exception as err:
        print("Exception occurred while attempting to run application. \n", err)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == "__main__":
    main()
