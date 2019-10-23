import numpy as np
import cv2
import glob
import os

from skimage import color, data, restoration


def rotate_images(path, resize_for_chart):
	try:
		path_extension = "/*.jpg"
		folder_path = "{}{}".format(path, path_extension)
		files = glob.glob(folder_path)
		
		for img in files:
			filename = img.split("\\")[-1]
			image = cv2.imread(img, cv2.IMREAD_COLOR)
			flipped_image = cv2.flip(image, 0)
			if(resize_for_chart):
				flipped_image = cv2.resize(flipped_image, dsize=(370, 340), interpolation=cv2.INTER_AREA)
			cv2.imwrite(img, flipped_image)
	
	except Exception as err:
		print(err)
	
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
	# picture_path = ''
	# backup_resize(picture_path)
	
	# picture_path_30 = ''
	# picture_path_25 = ''
	# picture_path_20 = ''
	# picture_path_10 = ''
	#
	# rotate_images(picture_path_30, True)
	# rotate_images(picture_path_25, True)
	# rotate_images(picture_path_20, True)
	# rotate_images(picture_path_10, True)
	# print("Finished resizing images")
