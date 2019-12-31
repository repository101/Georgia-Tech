#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the modules we're going to use
import numpy as np
import sys
import imageio
import os
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

from glob import glob


# In[2]:


# Define the function to display the optical flow
def display_flow(img, flow, stride=40, count=0):    
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1], (0,0,255), 
                            2, cv2.LINE_AA, 0, 0.2)
        
    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, 
                                  cv2.NORM_MINMAX)
    magnitude_normalized = ((norm_opt_flow / norm_opt_flow.max()) * 255)
    cv2.imshow('optical flow', img.astype(np.uint8))
    cv2.imwrite("Flows/frame{0:04d}.png".format(count), img)
    cv2.imshow('optical flow magnitude', magnitude_normalized.astype(np.uint8))
    cv2.imwrite("Flow_Magnitude/frame{0:04d}.png".format(count), magnitude_normalized)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test = np.hstack((gray_img, magnitude_normalized))
    cv2.imwrite("Combined/frame{0:04d}.png".format(count), test)

    k = cv2.waitKey(1)
    
    if k == 27:
        return 1
    else:
        return 0


# In[3]:
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


paths = getImageFileNames("C:/Users/Josh.Adams/OneDrive - Georgia Institute of Technology/"
                          "Georgia-Tech/CS 6475 - Computational Photography/assignments/"
                          "Final_Project/Combined")
resizeImages(paths, percent=0.35)
print()
# Open the video and grab its first frame. 
cap = cv2.VideoCapture("test.mp4")
_, prev_frame = cap.read()

# Next, read the frames one-by-one
# and compute the dense optical flow using Gunnar Farneback's algorithm. 
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)
init_flow = True

cnt=1
# Then, display the results
while True:
    status_cap, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if init_flow:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, opt_flow, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_USE_INITIAL_FLOW)
    
    prev_frame = np.copy(gray)
    cnt += 1
    if display_flow(frame, opt_flow, count=cnt):
        break;
    
    
cv2.destroyAllWindows()

