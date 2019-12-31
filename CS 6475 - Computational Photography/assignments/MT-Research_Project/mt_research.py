import numpy as np
import cv2
import os

from SeamCarver import *


def main():
	filename = "test.png"
	# filepath = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 64" \
	#            "75 - Computational Photography/assignments/MT-Research_Project/"
	filepath = "C:/Users/Josh.Adams/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/MT-Research_Project/"

	Carver = SeamCarver(name=filename, filepath=filepath,
						vertical=False,
						horizontal=False,
						scale_reduce=0.25,
						scale_expand=None,
						reduce=False,
						expand=False,
						forward_energy=True,
						useCV2=True)


if __name__ == "__main__":
	main()
