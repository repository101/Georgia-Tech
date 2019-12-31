import imageio
import os
import sys
import numpy as np
from glob import glob
import cv2


def readImages(image_dir):
    """This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))

    return image_files


def generateGif(files, alpha):
    for i in np.arange(0.01, 0.05, 0.01):
        i = np.round(i, 2)
        with imageio.get_writer("C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/TEST_Alpha_{}_Duration_{}sec.gif".format(alpha, i), mode="I", duration=i) as writer:
            for filename in files:
                image = imageio.imread(filename)
                writer.append_data(image)
    return


def getFramesFromVideo(pathToVideo, saveDirectory):
    # Source https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    # I did not make this, I used it to get the frames from the video that I captured from Youtube
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


def resizeFrames(files, percentOfOriginal, saveDirectory):
    saveDirectory += ("/Resized_" + str(percentOfOriginal) + "/")
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
    for i in files:
        newName = i.split("\\")
        newName = newName[-1]
        original = cv2.imread(i)
        resized = cv2.resize(original, None, fx=percentOfOriginal, fy=percentOfOriginal)
        new_path = saveDirectory + "/" + "resized_" + newName
        cv2.imwrite("{}".format(new_path), resized)
        print("{} Finished".format(new_path))
    print()
    return


def addCircles(image):
    # img_name = "/candle_diff3.png"
    img_name = "/Resized_0.5_diff3.png"
    temp_img = cv2.imread(image+img_name)
    new_temp = cv2.circle(temp_img, (70, 70), color=[0, 0, 255], radius=20, thickness=1,)
    cv2.imshow("TEMP", new_temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(image + "/WithCircle.png", new_temp)
    return



if __name__ == "__main__":
    directory = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/videos/source/Above_And_Beyond"
    output_directory_to_turn_to_gif = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/videos/out/Above_And_Beyond/Resized_0.1/"
    # output_directory_to_turn_to_gif = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/videos/out/candle/"
    video_frame_directory = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/videos/Video_Frames"
    video_path = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/videos/AB_Video_1.mp4"
    filenames = readImages(directory)
    output_files = readImages(output_directory_to_turn_to_gif)
    images = []
    alpha = 0.054
    generateGif(output_files, alpha=alpha)
    # getFramesFromVideo(pathToVideo=video_path, saveDirectory=directory)
    # resizeFrames(filenames, 0.10, saveDirectory=directory)
    # img = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/A6-Video_Textures/resources/Above_And_Beyond"
    # addCircles(img)
    print()
