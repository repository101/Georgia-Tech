import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import hdr as hdr

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
        images = [cv2.imread(path.join(IMG_FOLDER, "sample-00.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-01.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-02.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-03.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-04.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-05.png"))]

        if not all([im is not None for im in images]):
            raise IOError("Error, one or more sample images not found.")

        self.images = images
        self.exposures = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
                                     1 / 60.0, 1 / 40.0, 1 / 15.0])


if __name__ == '__main__':
    unittest.main()
