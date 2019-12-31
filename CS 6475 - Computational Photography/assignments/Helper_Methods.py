import numpy as np
import cv2
import glob
import os

# USE cv.filter2D for convolving over a matrix


def convolution_method(kernel, convolution_array=None):
	# Steps
	#   1) Flip kernel Top to Bottom
	#   2) Flip kernel Right to Left
	#   3) Apply Cross-Correlation
	kernel = np.flipud(kernel)
	kernel = np.fliplr(kernel)
	print(kernel)
	return


def cross_correlation(kernel, cross_correlation_array=None):
	return


def backup_resize(path):
	try:
		path_extension = "/*.jpg"
		backup_folder = "/backup_images"
		twenty_five_percent_folder = "/25_Percent"
		fifty_percent_folder = "/50_Percent"
		
		full_backup_directory = "{}{}".format(path, backup_folder)
		full_twenty_five_percent_directory = "{}{}".format(path, twenty_five_percent_folder)
		full_fifty_percent_directory = "{}{}".format(path, fifty_percent_folder)
		
		# Create BackUp Directory
		try:
			if not os.path.exists(full_backup_directory):
				os.mkdir(full_backup_directory)
		except OSError:
			print("Creation of the directory %s failed" % full_backup_directory)
		else:
			print("Successfully created the directory %s" % full_backup_directory)
		
		# Create 25 Percent Directory
		try:
			if not os.path.exists(full_twenty_five_percent_directory):
				os.mkdir(full_twenty_five_percent_directory)
		except OSError:
			print("Creation of the directory %s failed" % full_twenty_five_percent_directory)
		else:
			print("Successfully created the directory %s" % full_twenty_five_percent_directory)
		
		# Create 50 Percent Directory
		try:
			if not os.path.exists(full_fifty_percent_directory):
				os.mkdir(full_fifty_percent_directory)
		except OSError:
			print("Creation of the directory %s failed" % full_fifty_percent_directory)
		else:
			print("Successfully created the directory %s" % full_fifty_percent_directory)
		
		files = glob.glob("{}{}".format(path, path_extension))
		for img in files:
			# Get the actual file name
			filename = img.split('\\')[-1]
			
			# Read in the color data
			image_data = cv2.imread(img, cv2.IMREAD_COLOR)
			
			# Establish the path for the back up of the image
			backup_path = "{}{}{}".format(full_backup_directory, "/", filename)
			
			# Establish the path for the 25% scaled image
			twenty_five_path = "{}{}{}".format(full_twenty_five_percent_directory, "/25_", filename)
			
			# Establish the path for the 50% scaled image
			fifty_path = "{}{}{}".format(full_fifty_percent_directory, "/50_", filename)
			
			# Create back up of the original image
			cv2.imwrite(backup_path, image_data)
			
			# Create a 25% scaled image
			twenty_five_percent_image = cv2.resize(image_data, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
			
			# Save the 25% scaled image to the 25% folder
			cv2.imwrite(twenty_five_path, twenty_five_percent_image)

			# Create a 50% scaled image
			fifty_percent_image = cv2.resize(image_data, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
			
			# Save the 50% scaled image to the 50% folder
			cv2.imwrite(fifty_path, fifty_percent_image)
			
	except Exception as err:
		print(err)
	
	return


if __name__ == "__main__":
	# test_kernel = np.asarray([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']])
	# convolution_method(test_kernel)
	picture_path = 'C:/Users/joshu/OneDrive - Georgia Institute of Technology/Georgia-Tech/CS 6475 - Computational Photography/Course Assignments/A2-Camera_Obscura/Obscura_Pictures'
	backup_resize(picture_path)
	print("Finished resizing images")
