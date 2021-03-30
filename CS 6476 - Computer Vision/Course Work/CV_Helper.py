import os
import sys
import time

import numpy as np
import cv2


def normalize(img_in):
    img_out = np.zeros(img_in.shape)
    cv2.normalize(img_in, img_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_out


def select_gdir(gmag, gdir, mag_min, angle_low, angle_high):
    # Gradient Direction
    """
    :param gmag: (passed in should be the results from np.sqrt(gx**2 + gy**2))
    :param gdir: (passed in should be the results from np.arctan2(-gy, gx) * 180 / np.pi)
    :param mag_min:
    :param angle_low:
    :param angle_high:
    :return: gradient direction
    """
    result = gmag >= mag_min & angle_low <= gdir & gdir <= angle_high
    return result

