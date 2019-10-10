
import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import panorama as pano

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"


class Assignment8Test(unittest.TestCase):

    def setUp(self):
        images = [cv2.imread(path.join(IMG_FOLDER, "1.jpg")),
                  cv2.imread(path.join(IMG_FOLDER, "2.jpg")),
                  cv2.imread(path.join(IMG_FOLDER, "3.jpg"))]

        if any([im is None for im in images]):
            raise IOError("Error, one or more sample images not found.")

        self.images = images


if __name__ == '__main__':
    unittest.main()
