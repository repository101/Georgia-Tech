import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import blending as blend

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"


class Assignment6Test(unittest.TestCase):

    def setUp(self):
        self.black_img = cv2.imread(path.join(IMG_FOLDER, "black.jpg"))
        self.white_img = cv2.imread(path.join(IMG_FOLDER, "white.jpg"))
        self.mask_img = cv2.imread(path.join(IMG_FOLDER, "mask.jpg"))

        if any(map(lambda x: x is None,
                   [self.black_img, self.white_img, self.mask_img])):
            raise IOError("Error, samples image not found.")


if __name__ == '__main__':
    unittest.main()
