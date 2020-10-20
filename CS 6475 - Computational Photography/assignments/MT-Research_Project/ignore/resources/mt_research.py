import numpy as np
import cv2
import os


from SeamCarver import *


def main():
	filename = "fig8.png"
	# filepath = "C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 64" \
	#            "75 - Computational Photography/assignments/MT-Research_Project/"
	filepath = "C:/Users/Josh.Adams/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/assignments/MT-Research_Project/"

	Carver = SeamCarver(name=filename, filepath=filepath,
	                    vertical=True,
	                    horizontal=False,
	                    scale_reduce=None,
	                    scale_expand=0.5,
	                    reduce=False,
	                    expand=True,
	                    forward_energy=True)


if __name__ == "__main__":
	main()
