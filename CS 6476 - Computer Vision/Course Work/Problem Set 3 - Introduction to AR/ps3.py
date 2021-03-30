"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy import ndimage


def create_white_hsv_mask(hsv_img, other_img=None):
	# From my (Josh Adams) PS2 assignment CS 6476
	if other_img is None:
		lower = np.array([76, 0, 230])
		upper = np.array([114, 255, 255])
		return cv2.inRange(hsv_img, lower, upper)
	else:
		def nothing(x):
			pass
		
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		
		# create trackbars for color change
		
		cv2.createTrackbar('H Low', 'image', 0, 255, nothing)
		cv2.createTrackbar('H High', 'image', 180, 255, nothing)
		
		cv2.createTrackbar('S Low', 'image', 0, 255, nothing)
		cv2.createTrackbar('S High', 'image', 35, 255, nothing)
		
		cv2.createTrackbar('V Low', 'image', 230, 255, nothing)
		cv2.createTrackbar('V High', 'image', 255, 255, nothing)
		while 1:
			temp_image = np.copy(other_img)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
			
			# get current positions of four trackbars
			hl = cv2.getTrackbarPos('H Low', 'image')
			hh = cv2.getTrackbarPos('H High', 'image')
			sl = cv2.getTrackbarPos('S Low', 'image')
			sh = cv2.getTrackbarPos('S High', 'image')
			vl = cv2.getTrackbarPos('V Low', 'image')
			vh = cv2.getTrackbarPos('V High', 'image')
			
			lower = np.array([hl, sl, vl])
			upper = np.array([hh, sh, vh])
			
			mask = cv2.inRange(hsv_img, lower, upper)
			filtered_image_white = cv2.bitwise_and(temp_image, temp_image, mask=mask)
			cv2.imshow('image', filtered_image_white)
			cv2.imshow("Original", temp_image)
		
		cv2.destroyAllWindows()


def create_black_hsv_mask(hsv_img, other_img=None):
	if other_img is None:
		# From my (Josh Adams) PS2 assignment CS 6476
		lower = np.array([74, 18, 0])
		upper = np.array([133, 197, 110])
		return cv2.inRange(hsv_img, lower, upper)
	else:
		temp_gray = cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY)
		
		def nothing(x):
			pass
		
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		
		# create trackbars for color change
		
		cv2.createTrackbar('H Low', 'image', 74, 255, nothing)
		cv2.createTrackbar('H High', 'image', 133, 255, nothing)
		
		cv2.createTrackbar('S Low', 'image', 18, 255, nothing)
		cv2.createTrackbar('S High', 'image', 197, 255, nothing)
		
		cv2.createTrackbar('V Low', 'image', 0, 255, nothing)
		cv2.createTrackbar('V High', 'image', 110, 255, nothing)
		while 1:
			temp_image = np.copy(temp_gray)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
			
			# get current positions of four trackbars
			hl = cv2.getTrackbarPos('H Low', 'image')
			hh = cv2.getTrackbarPos('H High', 'image')
			sl = cv2.getTrackbarPos('S Low', 'image')
			sh = cv2.getTrackbarPos('S High', 'image')
			vl = cv2.getTrackbarPos('V Low', 'image')
			vh = cv2.getTrackbarPos('V High', 'image')
			
			lower = np.array([hl, sl, vl])
			upper = np.array([hh, sh, vh])
			
			mask = cv2.inRange(hsv_img, lower, upper)
			filtered_image_white = cv2.bitwise_and(temp_image, temp_image, mask=mask)
			cv2.imshow('image', filtered_image_white)
			cv2.imshow("Original", temp_image)
		
		cv2.destroyAllWindows()


def filter_points(points, is_simplistic=False):
	temp_points = np.copy(points)
	
	container = {}
	highest_count = None
	for i in range(len(temp_points)):
		temp_distance = np.round(
			np.sqrt((temp_points[i][0] - temp_points[:, 0]) ** 2 + (temp_points[i][1] - temp_points[:, 1]) ** 2), 3)
		temp_distance[i] = np.nan
		key = f"({temp_points[i][0]},{temp_points[i][1]})"
		if key not in container:
			container[key] = 1
		else:
			if is_simplistic:
				highest_count = (temp_points[i][0] - 1, temp_points[i][1] - 1)
			else:
				highest_count = (temp_points[i][0], temp_points[i][1])
			container[key] += 1
	if highest_count is not None:
		return highest_count
	else:
		t = max(container, key=container.get)
		t = t.replace("(", "")
		t = t.replace(")", "")
		vals = t.split(",")
		if is_simplistic:
			return int(vals[0]) - 1, int(vals[1]) - 1
		else:
			return int(vals[0]), int(vals[1])


def check_results(points0, points1, is_noisy=False):
	results = []
	if is_noisy:
		# When noisy Points0 will take precedence over Points 1, unless similar points are
		#   within threshold of distance from each other
		for pt0, pt1 in zip(points0, points1):
			distance = euclidean_distance(pt0, pt1)
			total_allowable_distance = 4
			dst_pct = distance / total_allowable_distance
			
			pt0_pct = 0.5 - (dst_pct * 0.5)
			pt1_pct = (0.5 - pt0_pct) + 0.5
			if np.all(distance > total_allowable_distance):
				dst_pct = dst_pct % 1
				if distance > 3 * total_allowable_distance:
					pt0_pct = 1.0
					pt1_pct = 0.0
					# When the distances are close we will take a weighted average where Points1 are more accurate
					results.append((int(np.ceil(pt0[0] * pt0_pct + pt1[0] * pt1_pct)),
									int(np.ceil(pt0[1] * pt0_pct + pt1[1] * pt1_pct))))
				else:
					if distance > 2 * total_allowable_distance:
						dst_pct = 0.00
					pt0_pct = 0.5 - (dst_pct * 0.5)
					pt1_pct = (0.5 - pt0_pct) + 0.5
					# When the distances are close we will take a weighted average where Points1 are more accurate
					results.append((int(np.ceil(pt0[0] * pt0_pct + pt1[0] * pt1_pct)),
									int(np.ceil(pt0[1] * pt0_pct + pt1[1] * pt1_pct))))
			else:
				# When the distances are close we will take a weighted average where Points1 are more accurate
				results.append((int(np.ceil(pt0[0] * pt0_pct + pt1[0] * pt1_pct)),
								int(np.ceil(pt0[1] * pt0_pct + pt1[1] * pt1_pct))))
		return results
	
	else:
		for pt0, pt1 in zip(points0, points1):
			distance = euclidean_distance(pt0, pt1)
			total_allowable_distance = 5
			dst_pct = distance / total_allowable_distance
			
			pt0_pct = 0.5 - (dst_pct * 0.5)
			pt1_pct = (0.5 - pt0_pct) + 0.5
			if np.all(distance > total_allowable_distance):
				pt0_pct = 0.0
				pt1_pct = 1.0
			
			results.append(
				(int(np.ceil(pt0[0] * pt0_pct + pt1[0] * pt1_pct)), int(np.ceil(pt0[1] * pt0_pct + pt1[1] * pt1_pct))))
	return results


def calc_angle_in_degrees(pt1, pt2):
	# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
	return np.round(np.rad2deg(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))


def find_lines(image, other_image=None, testing=False):
	if not testing:
		if other_image is not None:
			if len(image.shape) > 2:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			temp_image = np.copy(other_image)
			accumulator = np.zeros(shape=image.shape[:2])
			lines = cv2.HoughLines(image, rho=1.0, theta=np.pi / 180, threshold=10, srn=0, stn=0)
			if lines is not None:
				for i in range(0, len(lines)):
					temp_accumulator = np.zeros(shape=image.shape[:2])
					rho = lines[i][0][0]
					theta = lines[i][0][1]
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a * rho
					y0 = b * rho
					pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
					pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
					cv2.line(temp_accumulator, pt1=pt1, pt2=pt2, color=(1), thickness=1)
					accumulator += temp_accumulator
			return lines, accumulator
	else:
		def nothing(x):
			pass
		display_image = np.copy(image)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		# cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
		# create trackbars for color change
		
		cv2.createTrackbar('RHO', 'image', 10, 100, nothing)
		cv2.createTrackbar('Threshold', 'image', 50, 500, nothing)
		cv2.createTrackbar('Canny Threshold 1', 'image', 50, 250, nothing)
		cv2.createTrackbar('Canny Threshold 2', 'image', 100, 500, nothing)
		cv2.createTrackbar('Degree', 'image', 1, 180, nothing)
		# cv2.createTrackbar('P Lines Thresh', 'image', 10, 100, nothing)
		# cv2.createTrackbar('Min Line Length', 'image', 10, 50, nothing)
		# cv2.createTrackbar('Max Line Gap', 'image', 5, 50, nothing)
		while (1):
			temp_image = np.copy(gray)
			temp_image = cv2.bilateralFilter(temp_image, 3, 75, 75)
			temp_disp = np.copy(display_image)
			# temp_image = cv2.medianBlur(temp_image, 7)
			# temp_image = cv2.morphologyEx(image, cv2.MORPH_RECT, np.ones((3, 3), np.uint8))
			# temp_image = cv2.filter2D(temp_image, -1, np.ones((3, 3), np.float32) / 9)
			# temp_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
			# temp_image = cv2.bilateralFilter(temp_image, 9, 75, 75)
			color = np.copy(other_image)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
			
			# get current positions of four trackbars
			rho = cv2.getTrackbarPos('RHO', 'image')
			threshold = cv2.getTrackbarPos('Threshold', 'image')
			canny_1 = cv2.getTrackbarPos('Canny Threshold 1', 'image')
			canny_2 = cv2.getTrackbarPos('Canny Threshold 2', 'image')
			deg = cv2.getTrackbarPos('Degree', 'image')
			# lines_thresh = cv2.getTrackbarPos('P Lines Thresh', 'image')
			# line_length = cv2.getTrackbarPos('Min Line Length', 'image')
			# line_gap = cv2.getTrackbarPos('Max Line Gap', 'image')
			# # if rho > 0 and threshold > 1:

			if rho > 0 and threshold > 1 and deg > 0:
				temp_rho = 0.1 * rho
				temp_color = np.copy(color)
				accumulator = np.zeros(shape=color.shape[:2])
				edges = cv2.Canny(temp_image, apertureSize=3, threshold1=canny_1, threshold2=canny_2)
				# linesP = cv2.HoughLinesP(temp_image, temp_rho, (np.pi / 180) * 30, lines_thresh, None, line_length, line_gap)
				
				# if linesP is not None:
				# 	for i in range(0, len(linesP)):
				# 		temp_accumulator = np.zeros(shape=color.shape[:2])
				# 		l = linesP[i][0]
				# 		cv2.line(temp_accumulator, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
				# 		accumulator += temp_accumulator
						
				lines = cv2.HoughLines(edges, rho=temp_rho, theta=(np.pi / 180) * deg, threshold=threshold, srn=0, stn=0)
				if lines is not None:
					for i in range(0, len(lines)):
						temp_accumulator = np.zeros(shape=color.shape[:2])
						rho = lines[i][0][0]
						theta = lines[i][0][1]
						a = np.cos(theta)
						b = np.sin(theta)
						x0 = a * rho
						y0 = b * rho
						pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
						pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
						cv2.line(temp_accumulator, pt1=pt1, pt2=pt2, color=1, thickness=1, lineType=cv2.LINE_AA)
						# cv2.line(temp_accumulator, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
						accumulator += temp_accumulator
					cv2.imshow('edges', edges)
					max_location = np.where(accumulator >= accumulator.max() * 0.90)
					if max_location is not None:
						if len(max_location) > 0:
							t = np.vstack((max_location[0], max_location[1])).T
							for loc in t:
								cv2.circle(img=temp_color, center=(loc[1], loc[0]), radius=3, color=(0,255,0),
										   thickness=-1, lineType=cv2.LINE_AA)
					cv2.imshow('Display', temp_color)
					
					cv2.putText(img=accumulator, text=f"max val: {accumulator.max():.3f}", org=(300, 100), color=255,
								fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								thickness=2, fontScale=1)
				
					accumulator[accumulator < 2] = 0
					cv2.imshow('image', accumulator)
					# val_count = np.unique(accumulator, return_counts=True)
					# if accumulator.max() > 2:
					# 	max_location = np.where(accumulator == accumulator.max())
					# 	if max_location is not None:
					# 		if len(max_location) > 0:
					# 			t = np.vstack((max_location[0], max_location[1])).T
					# 			for loc in t:
					# 				cv2.circle(img=color, center=(loc[1], loc[0]), radius=3, color=(0, 255, 0),
					# 						   thickness=-1)
					
					
		
		cv2.destroyAllWindows()


def find_extrema(pts, is_simplistic=False, swap_vals=False):
	points = np.copy(pts).astype(np.float)
	if points[0].shape[0] == 3:
		temp_val = np.copy(points[points[:, 0] == np.nanmin(points[:, 0])][0])
		idx = np.where(points == temp_val)
		temp_count = np.unique(idx[0], return_counts=True)
		temp_idx = temp_count[0][np.argmax(temp_count[1])]
		points[temp_idx] = np.asarray([np.nan, np.nan, np.nan])
		next_val = np.copy(points[np.nanargmin(points[:, 0] - temp_val[0])])
		idx = np.where(points == next_val)
		points[idx[0][0]] = np.asarray([np.nan, np.nan, np.nan])
		temp_val = temp_val.astype(np.int)
		next_val = next_val.astype(np.int)
		if is_simplistic:
			if temp_val[1] < next_val[1]:
				top_left = (temp_val[1], temp_val[0])
				bottom_left = (next_val[1], next_val[0])
			else:
				top_left = (next_val[1], next_val[0])
				bottom_left = (temp_val[1], temp_val[0])
		else:
			if temp_val[1] < next_val[1]:
				top_left = (temp_val[0], temp_val[1])
				bottom_left = (next_val[0], next_val[1])
			else:
				top_left = (next_val[0], next_val[1])
				bottom_left = (temp_val[0], temp_val[1])
		
		temp_val = np.copy(points[points[:, 1] == np.nanmin(points[:, 1])][0])
		idx = np.where(points == temp_val)
		temp_count = np.unique(idx[0], return_counts=True)
		temp_idx = temp_count[0][np.argmax(temp_count[1])]
		points[temp_idx] = np.asarray([np.nan, np.nan, np.nan])
		next_val = np.copy(points[np.nanargmin(points[:, 1] - temp_val[1])])
		idx = np.where(points == next_val)
		points[idx[0][0]] = np.asarray([np.nan, np.nan, np.nan])
		temp_val = temp_val.astype(np.int)
		next_val = next_val.astype(np.int)
		if is_simplistic:

			if temp_val[1] < next_val[1]:
				top_right = (temp_val[1], temp_val[0])
				bottom_right = (next_val[1], next_val[0])
			else:
				top_right = (next_val[1], next_val[0])
				bottom_right = (temp_val[1], temp_val[0])
		else:
			if temp_val[1] < next_val[1]:
				top_right = (temp_val[0], temp_val[1])
				bottom_right = (next_val[0], next_val[1])
			else:
				top_right = (next_val[0], next_val[1])
				bottom_right = (temp_val[0], temp_val[1])
		if swap_vals:
			return [(top_left[1], top_left[0]), (bottom_left[1], bottom_left[0]),
					(top_right[1], top_right[0]), (bottom_right[1], bottom_right[0])]
		else:
			return [top_left, bottom_left, top_right, bottom_right]
	elif points[0].shape[0] == 2:
		temp_val = np.copy(points[points[:, 1] == np.nanmin(points[:, 1])][0])
		for i in range(len(points)):
			if np.all(points[i] == temp_val):
				points[i] = np.asarray([np.nan, np.nan])
				break
		next_val = np.copy(points[np.nanargmin(points[:, 1] - temp_val[0])])
		for i in range(len(points)):
			if np.all(points[i] == next_val):
				points[i] = np.asarray([np.nan, np.nan])
				break
		temp_val = temp_val.astype(np.int)
		next_val = next_val.astype(np.int)
		if is_simplistic:
			if temp_val[0] < next_val[0]:
				top_left = (temp_val[1], temp_val[0])
				bottom_left = (next_val[1], next_val[0])
			else:
				top_left = (next_val[1], next_val[0])
				bottom_left = (temp_val[1], temp_val[0])
		else:
			if temp_val[0] < next_val[0]:
				top_left = (temp_val[0], temp_val[1])
				bottom_left = (next_val[0], next_val[1])
			else:
				top_left = (next_val[0], next_val[1])
				bottom_left = (temp_val[0], temp_val[1])
		
		temp_val = np.copy(points[points[:, 0] == np.nanmin(points[:, 0])][0])
		for i in range(len(points)):
			if np.all(points[i] == temp_val):
				points[i] = np.asarray([np.nan, np.nan])
				break
		next_val = np.copy(points[np.nanargmin(points[:, 0] - temp_val[1])])
		for i in range(len(points)):
			if np.all(points[i] == next_val):
				points[i] = np.asarray([np.nan, np.nan])
				break
		temp_val = temp_val.astype(np.int)
		next_val = next_val.astype(np.int)
		if is_simplistic:
			if temp_val[0] < next_val[0]:
				top_right = (temp_val[1], temp_val[0])
				bottom_right = (next_val[1], next_val[0])
			else:
				top_right = (next_val[1], next_val[0])
				bottom_right = (temp_val[1], temp_val[0])
		else:
			if temp_val[1] < next_val[1]:
				top_right = (temp_val[0], temp_val[1])
				bottom_right = (next_val[0], next_val[1])
			elif temp_val[0] < next_val[0]:
				bottom_right = (next_val[0], next_val[1])
				top_right = (temp_val[0], temp_val[1])
			else:
				top_right = (next_val[0], next_val[1])
				bottom_right = (temp_val[0], temp_val[1])
		if swap_vals:
			return [(top_left[1], top_left[0]), (bottom_left[1], bottom_left[0]),
					(top_right[1], top_right[0]), (bottom_right[1], bottom_right[0])]
		else:
			return [top_left, bottom_left, top_right, bottom_right]


def run_simplistic(image):
	temp_image = np.copy(image)
	grayscale = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
	corners = find_corners(grayscale=grayscale, is_simplistic=True)
	temp_results = find_extrema(np.copy(corners), is_simplistic=True)
	return [(int(i[0]), int(i[1])) for i in temp_results]


def find_circles(grayscale, other_img=None, is_noisy=False, is_simplistic=False):
	if other_img is None:
		if is_noisy:
			mask = np.zeros(shape=grayscale.shape)
			edges = cv2.Canny(grayscale, 100, 430, None, 3)
			circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100, param1=20, param2=5,
									   minRadius=10, maxRadius=100)
			if circles is not None:
				if len(circles[0]) > 0:
					
					if len(circles[0]) > 4:
						circles = circles[0][:4]
						for i in circles:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=255,
									   thickness=-1)
					else:
						for i in circles[0]:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=255,
									   thickness=-1)
							
			return circles, mask
		
		elif is_simplistic:
			mask = np.zeros(shape=grayscale.shape)
			temp_image = np.copy(grayscale)
			edges = cv2.Canny(temp_image, 20, 20, None, 3)
			circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=50, param1=5, param2=5,
									   minRadius=5, maxRadius=100)
			if circles is not None:
				if len(circles[0]) > 0:
					
					if len(circles[0]) > 4:
						circles = circles[0][:4]
						for i in circles:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=(255),
									   thickness=-1)
					else:
						for i in circles[0]:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=(255),
									   thickness=-1)
			
			return circles, mask
		else:
			mask = np.zeros(shape=grayscale.shape)
			edges = cv2.Canny(grayscale, 104, 229, None, 3)
			circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100, param1=20, param2=16,
									   minRadius=12, maxRadius=100)
			if circles is not None:
				if len(circles[0]) > 0:
					
					if len(circles[0]) > 4:
						circles = circles[0][:4]
						for i in circles:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=(255),
									   thickness=-1)
					else:
						for i in circles[0]:
							cv2.circle(mask, center=(i[0], i[1]), radius=i[2], color=(255),
									   thickness=-1)
			
			return circles, mask
	else:
		# Only get here during testing
		temp_gray = np.copy(grayscale)
		temp_img = np.copy(other_img)
		
		def nothing(x):
			pass
		
		if is_noisy:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('DP', 'image', 10, 10, nothing)
			cv2.createTrackbar('MinDist', 'image', 100, 200, nothing)
			cv2.createTrackbar('MinRadius', 'image', 20, 100, nothing)
			cv2.createTrackbar('MaxRadius', 'image', 100, 100, nothing)
			cv2.createTrackbar('Param1', 'image', 20, 100, nothing)
			cv2.createTrackbar('Param2', 'image', 5, 100, nothing)
			cv2.createTrackbar('Canny Thresh 1', 'image', 100, 1000, nothing)
			cv2.createTrackbar('Canny Thresh 2', 'image', 430, 1000, nothing)
			#
			# # Only used for testing
			# known_cords = np.asarray([[54, 205],
			#                           [101, 574],
			#                           [943, 199],
			#                           [902, 578]])
		
		elif is_simplistic:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('DP', 'image', 10, 10, nothing)
			cv2.createTrackbar('MinDist', 'image', 50, 1000, nothing)
			cv2.createTrackbar('MinRadius', 'image', 5, 100, nothing)
			cv2.createTrackbar('MaxRadius', 'image', 100, 100, nothing)
			cv2.createTrackbar('Param1', 'image', 5, 30, nothing)
			cv2.createTrackbar('Param2', 'image', 5, 30, nothing)
			cv2.createTrackbar('Canny Thresh 1', 'image', 20, 1000, nothing)
			cv2.createTrackbar('Canny Thresh 2', 'image', 20, 1000, nothing)
			# Only used for testing
			# known_cords = np.asarray([[114, 40],
			#                           [151, 142],
			#                           [355, 112],
			#                           [346, 49]])
		else:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('DP', 'image', 10, 10, nothing)
			cv2.createTrackbar('MinDist', 'image', 100, 1000, nothing)
			cv2.createTrackbar('MinRadius', 'image', 10, 100, nothing)
			cv2.createTrackbar('MaxRadius', 'image', 100, 100, nothing)
			cv2.createTrackbar('Param1', 'image', 20, 100, nothing)
			cv2.createTrackbar('Param2', 'image', 20, 100, nothing)
			cv2.createTrackbar('Canny Thresh 1', 'image', 104, 1000, nothing)
			cv2.createTrackbar('Canny Thresh 2', 'image', 229, 1000, nothing)
			# Only used for testing
			# known_cords = np.asarray([[106, 678],
			#                           [100, 678],
			#                           [933, 558],
			#                           [944, 208]])
		
		if is_noisy:
			known_cords = np.asarray([[54, 205],
									  [101, 574],
									  [943, 199],
									  [902, 578]])
		elif is_simplistic:
			known_cords = np.asarray([[114, 40],
									  [151, 142],
									  [355, 112],
									  [346, 49]])
		else:
			known_cords = np.asarray([[106, 678],
									  [100, 678],
									  [933, 558],
									  [944, 208]])
		
		# best_distance = np.inf
		#
		# val = 0
		# for pt in known_cords:
		# 	temp_val = np.nanmin(np.sqrt((pt[0] - known_cords[:, 0]) ** 2 + (pt[1] - known_cords[:, 1]) ** 2))
		# 	val += temp_val
		#
		# best_possible = 8582.725648816971
		# best_vals = None
		# start_time = time.time()
		# min_dist = 10
		# # min_radius = 5
		#
		# for min_radius in range(5, 20, 1):
		# 	for param_2 in range(10, 20, 1):
		# 		for canny_thresh_1 in range(1, 50, 1):
		# 			for canny_thresh_2 in range(10, 25, 1):
		# 				edges = cv2.Canny(temp_gray, canny_thresh_1, canny_thresh_2, None, 3)
		# 				circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_dist, param1=20,
		# 				                           param2=param_2,
		# 				                           minRadius=min_radius, maxRadius=100)
		# 				if circles is not None:
		# 					if len(circles[0]) > 0:
		# 						total_dist = 0
		# 						if len(circles[0]) >= 4:
		# 							temp_circles = circles[0][:4]
		# 							for pt in known_cords:
		# 								temp_val = np.nanmin(
		# 									np.sqrt(
		# 										(pt[0] - temp_circles[:, 0]) ** 2 + (pt[1] - temp_circles[:, 1]) ** 2))
		# 								total_dist += temp_val
		# 							if np.round(total_dist, 4) < best_distance:
		# 								best_distance = np.round(total_dist, 4)
		# 								best_vals = {"DP": 1.0, "MinDist": min_dist, "MinRadius": min_radius,
		# 								             "MaxRadius": 100, "Param 1": 20, "Param 2": param_2,
		# 								             "CannyThreshold 1": canny_thresh_1,
		# 								             "CannyThreshold 2": canny_thresh_2
		# 								             }
		# 								print(f"\n\t Best Distance: {best_distance}")
		# 								print(f"\n\tBest Parameters: \n\t{best_vals}")
		# end_time = time.time()
		# elapsed_time = end_time - start_time
		# print(f"\n\tTotal Elapsed Time: {elapsed_time:.3f}s")
		
		while 1:
			# temp_image = np.copy(cv2.bilateralFilter(grayscale, 9, 75, 75))
			# temp_image = np.copy(cv2.bilateralFilter(cv2.medianBlur(grayscale, 3), 9, 75, 75))
			temp_image = np.copy(grayscale)
			temp_img_1 = np.copy(temp_img)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
			if is_noisy:
				# get current positions of four trackbars
				dp = cv2.getTrackbarPos('DP', 'image')
				min_dist = cv2.getTrackbarPos('MinDist', 'image')
				min_radius = cv2.getTrackbarPos('MinRadius', 'image')
				max_radius = cv2.getTrackbarPos('MaxRadius', 'image')
				param_1 = cv2.getTrackbarPos('Param1', 'image')
				param_2 = cv2.getTrackbarPos('Param2', 'image')
				canny_thresh_1 = cv2.getTrackbarPos('Canny Thresh 1', 'image')
				canny_thresh_2 = cv2.getTrackbarPos('Canny Thresh 2', 'image')
				
				edges = cv2.Canny(temp_gray, canny_thresh_1, canny_thresh_2, None, 3)
				if min_dist > 0 and min_radius > 0 and max_radius > 0 and param_1 > 0 and param_2 > 0 and dp > 0:
					temp_dp = np.round(0.1 * dp, 2)
					circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=temp_dp, minDist=min_dist, param1=param_1,
											   param2=param_2,
											   minRadius=min_radius, maxRadius=max_radius)
					
					if circles is not None:
						if len(circles[0]) > 0:
							
							if len(circles[0]) > 4:
								circles = circles[0][:4]
								for i in circles:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(0, 255, 0),
											   thickness=2)
							else:
								for i in circles[0]:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(0, 255, 0),
											   thickness=2)
					cv2.putText(temp_img_1, f"DP: {temp_dp}", org=(temp_image.shape[1] // 3, 50), color=(0, 0, 0),
								thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2)
				cv2.imshow("Edges", edges)
				cv2.imshow('image', temp_img_1)
			elif is_simplistic:
				# get current positions of four trackbars
				dp = cv2.getTrackbarPos('DP', 'image')
				min_dist = cv2.getTrackbarPos('MinDist', 'image')
				min_radius = cv2.getTrackbarPos('MinRadius', 'image')
				max_radius = cv2.getTrackbarPos('MaxRadius', 'image')
				param_1 = cv2.getTrackbarPos('Param1', 'image')
				param_2 = cv2.getTrackbarPos('Param2', 'image')
				canny_thresh_1 = cv2.getTrackbarPos('Canny Thresh 1', 'image')
				canny_thresh_2 = cv2.getTrackbarPos('Canny Thresh 2', 'image')
				
				edges = cv2.Canny(temp_image, canny_thresh_1, canny_thresh_2, None, 3)
				if min_dist > 0 and min_radius > 0 and max_radius > 0 and param_1 > 0 and param_2 > 0 and dp > 0:
					temp_dp = np.round(0.1 * dp, 2)
					circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=temp_dp, minDist=min_dist, param1=param_1,
											   param2=param_2,
											   minRadius=min_radius, maxRadius=max_radius)
					total_dist = 0
					if circles is not None:
						if len(circles[0]) > 0:
							if len(circles[0]) > 200:
								circles = circles[0][:200]
								for i in circles:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(0, 255, 0),
											   thickness=2)
							else:
								
								for i in circles[0][:4]:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(255, 0, 255),
											   thickness=1)
								for pt in known_cords:
									temp_val = np.round(np.nanmin(np.sqrt(
										(pt[0] - circles[0][:4][:, 0]) ** 2 + (pt[1] - circles[0][:4][:, 1]) ** 2)), 3)
									total_dist += temp_val
					cv2.putText(temp_img_1, f"Distance: {total_dist:.3f}", org=(temp_image.shape[1] // 3, 20),
								color=(0, 0, 0),
								thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								fontScale=0.75)
			else:
				# get current positions of four trackbars
				dp = cv2.getTrackbarPos('DP', 'image')
				min_dist = cv2.getTrackbarPos('MinDist', 'image')
				min_radius = cv2.getTrackbarPos('MinRadius', 'image')
				max_radius = cv2.getTrackbarPos('MaxRadius', 'image')
				param_1 = cv2.getTrackbarPos('Param1', 'image')
				param_2 = cv2.getTrackbarPos('Param2', 'image')
				canny_thresh_1 = cv2.getTrackbarPos('Canny Thresh 1', 'image')
				canny_thresh_2 = cv2.getTrackbarPos('Canny Thresh 2', 'image')
				
				edges = cv2.Canny(temp_image, canny_thresh_1, canny_thresh_2, None, 3)
				if min_dist > 0 and min_radius > 0 and max_radius > 0 and param_1 > 0 and param_2 > 0 and dp > 0:
					temp_dp = np.round(0.1 * dp, 2)
					circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=temp_dp, minDist=min_dist,
											   param1=param_1,
											   param2=param_2,
											   minRadius=min_radius, maxRadius=max_radius)
					total_dist = 0
					if circles is not None:
						if len(circles[0]) > 0:
							if len(circles[0]) > 200:
								circles = circles[0][:200]
								for i in circles:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(0, 255, 0),
											   thickness=2)
							else:
								# circles = circles[0][:len(circles)]
								for i in circles[0][:4]:
									cv2.circle(temp_img_1, center=(i[0], i[1]), radius=i[2], color=(255, 0, 255),
											   thickness=2)
								for pt in known_cords:
									temp_val = np.round(np.nanmin(np.sqrt(
										(pt[0] - circles[0][:4][:, 0]) ** 2 + (pt[1] - circles[0][:4][:, 1]) ** 2)),
										3)
									total_dist += temp_val
					# cv2.putText(temp_img_1, f"DP: {temp_dp}", org=(temp_image.shape[1] // 3, 50), color=(0, 0, 0),
					#             thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					#             fontScale=2)
					cv2.putText(temp_img_1, f"Distance: {total_dist:.3f}", org=(temp_image.shape[1] // 3, 20),
								color=(0, 0, 0),
								thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								fontScale=1.0)
				cv2.imshow("Edges", edges)
				cv2.imshow('image', temp_img_1)
		
		cv2.destroyAllWindows()


def find_corners(grayscale, other_img=None, is_noisy=False, is_simplistic=False):
	if other_img is None:
		if is_noisy:
			temp_image = np.copy(grayscale)
			# corners = cv2.cornerHarris(temp_image, 10, 5, np.round(0.01 * 10, 2))
			corners = cv2.cornerHarris(temp_image, blockSize=20, ksize=21, k=np.round(0.01 * 7, 2))
			new_corners = np.zeros(shape=temp_image.shape)
			# t = new_corners[corners > 1]
			new_corners[corners > 0] = (corners[corners > 0] / corners.max()) * 255.0
			new_corners[new_corners < 50] = 0
			
			filtered_points = np.where(new_corners > 1)
			if len(filtered_points[0]) > 1:
				corner_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
				corner_clusters = cv2.kmeans(corner_points, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
				if corner_clusters is not None:
					if len(corner_clusters) > 1:
						if corner_clusters[2] is not None:
							if len(corner_clusters[2]) > 0:
								corner_cluster_centers = np.round(corner_clusters[2])
								temp_clusters = np.ravel(corner_cluster_centers)
								cluster_result = np.unique(temp_clusters, return_counts=True)
								new_corner_clusters = []
								for cluster_idx in range(len(corner_cluster_centers)):
									temp_point = []
									x_val = corner_cluster_centers[cluster_idx][0]
									y_val = corner_cluster_centers[cluster_idx][1]
									x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
									y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
									if x_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 0] - x_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_x = (alt_val[0] + x_val) // 2
										temp_point.append(new_x)
									else:
										temp_point.append(x_val)
									
									if y_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 1] - y_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_y = (alt_val[1] + y_val) // 2
										temp_point.append(new_y)
									else:
										temp_point.append(y_val)
									new_corner_clusters.append(temp_point)
			# cv2.imshow("img", new_corners)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			return new_corners
		
		elif is_simplistic:
			temp_image = np.copy(grayscale)
			corners = cv2.cornerHarris(temp_image, 12, 9, np.round(0.01 * 12, 2))
			new_corners = np.zeros(shape=temp_image.shape)
			# t = new_corners[corners > 1]
			new_corners[corners > 0] = (corners[corners > 0] / corners.max()) * 255.0
			new_corners[new_corners < 50] = 0
			
			filtered_points = np.where(new_corners > 1)
			if len(filtered_points[0]) > 1:
				corner_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
				corner_clusters = cv2.kmeans(corner_points, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
				if corner_clusters is not None:
					if len(corner_clusters) > 1:
						if corner_clusters[2] is not None:
							if len(corner_clusters[2]) > 0:
								corner_cluster_centers = np.round(corner_clusters[2])
								temp_clusters = np.ravel(corner_cluster_centers)
								cluster_result = np.unique(temp_clusters, return_counts=True)
								new_corner_clusters = []
								for cluster_idx in range(len(corner_cluster_centers)):
									temp_point = []
									x_val = corner_cluster_centers[cluster_idx][0]
									y_val = corner_cluster_centers[cluster_idx][1]
									x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
									y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
									if x_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 0] - x_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_x = (alt_val[0] + x_val) // 2
										temp_point.append(new_x)
									else:
										temp_point.append(x_val)
									
									if y_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 1] - y_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_y = (alt_val[1] + y_val) // 2
										temp_point.append(new_y)
									else:
										temp_point.append(y_val)
									new_corner_clusters.append(temp_point)
								return np.asarray(corner_cluster_centers)
		else:
			temp_image = np.copy(grayscale)
			corners = cv2.cornerHarris(temp_image, 10, 5, np.round(0.01 * 10, 2))
			new_corners = np.zeros(shape=temp_image.shape)
			# t = new_corners[corners > 1]
			new_corners[corners > 0] = (corners[corners > 0] / corners.max()) * 255.0
			new_corners[new_corners < 50] = 0
			
			filtered_points = np.where(new_corners > 1)
			if len(filtered_points[0]) > 1:
				corner_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
				corner_clusters = cv2.kmeans(corner_points, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
				if corner_clusters is not None:
					if len(corner_clusters) > 1:
						if corner_clusters[2] is not None:
							if len(corner_clusters[2]) > 0:
								corner_cluster_centers = np.round(corner_clusters[2])
								temp_clusters = np.ravel(corner_cluster_centers)
								cluster_result = np.unique(temp_clusters, return_counts=True)
								new_corner_clusters = []
								for cluster_idx in range(len(corner_cluster_centers)):
									temp_point = []
									x_val = corner_cluster_centers[cluster_idx][0]
									y_val = corner_cluster_centers[cluster_idx][1]
									x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
									y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
									if x_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 0] - x_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_x = (alt_val[0] + x_val) // 2
										temp_point.append(new_x)
									else:
										temp_point.append(x_val)
									
									if y_val_count == 1:
										diff = np.abs(corner_cluster_centers[:, 1] - y_val)
										temp_diff = diff == diff[diff > 0].min()
										alt_val = corner_cluster_centers[temp_diff, :][0]
										new_y = (alt_val[1] + y_val) // 2
										temp_point.append(new_y)
									else:
										temp_point.append(y_val)
									new_corner_clusters.append(temp_point)
			return new_corners
	else:
		# Only get here during testing
		temp_img = np.copy(other_img)
		
		def nothing(x):
			pass
		
		if is_noisy:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('BlockSize', 'image', 20, 30, nothing)
			cv2.createTrackbar('KSize', 'image', 21, 30, nothing)
			cv2.createTrackbar('KFree', 'image', 29, 100, nothing)
		
		elif is_simplistic:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('BlockSize', 'image', 12, 30, nothing)
			cv2.createTrackbar('KSize', 'image', 9, 30, nothing)
			cv2.createTrackbar('KFree', 'image', 12, 100, nothing)
		
		else:
			# create trackbars for color change
			cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
			cv2.createTrackbar('BlockSize', 'image', 10, 30, nothing)
			cv2.createTrackbar('KSize', 'image', 7, 30, nothing)
			cv2.createTrackbar('KFree', 'image', 12, 100, nothing)
		
		while 1:
			temp_img_1 = np.copy(temp_img)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
			
			# get current positions of four trackbars
			block_size = cv2.getTrackbarPos('BlockSize', 'image')
			k_size = cv2.getTrackbarPos('KSize', 'image')
			k_free = cv2.getTrackbarPos('KFree', 'image')
			
			if block_size > 0 and k_size > 0 and k_free > 0 and k_size % 2 == 1 and k_size < 31 and block_size < 31:
				temp_k_free = np.round(0.01 * k_free, 2)
				corners = cv2.cornerHarris(grayscale, block_size, k_size, temp_k_free)
				new_corners = np.zeros(shape=grayscale.shape)
				# t = new_corners[corners > 1]
				new_corners[corners > 0] = (corners[corners > 0] / corners.max()) * 255.0
				new_corners[new_corners < 50] = 0
				
				filtered_points = np.where(new_corners > 1)
				if len(filtered_points[0]) > 1:
					corner_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
					corner_cluster_centers = None
					criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
					corner_clusters = cv2.kmeans(corner_points, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
					if corner_clusters is not None:
						if len(corner_clusters) > 1:
							if corner_clusters[2] is not None:
								if len(corner_clusters[2]) > 0:
									corner_cluster_centers = np.round(corner_clusters[2])
									temp_clusters = np.ravel(corner_cluster_centers)
									cluster_result = np.unique(temp_clusters, return_counts=True)
									new_corner_clusters = []
									for cluster_idx in range(len(corner_cluster_centers)):
										temp_point = []
										x_val = corner_cluster_centers[cluster_idx][0]
										y_val = corner_cluster_centers[cluster_idx][1]
										x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
										y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
										if x_val_count == 1:
											diff = np.abs(corner_cluster_centers[:, 0] - x_val)
											temp_diff = diff == diff[diff > 0].min()
											alt_val = corner_cluster_centers[temp_diff, :][0]
											new_x = (alt_val[0] + x_val) // 2
											temp_point.append(new_x)
										else:
											temp_point.append(x_val)
										
										if y_val_count == 1:
											diff = np.abs(corner_cluster_centers[:, 1] - y_val)
											temp_diff = diff == diff[diff > 0].min()
											alt_val = corner_cluster_centers[temp_diff, :][0]
											new_y = (alt_val[1] + y_val) // 2
											temp_point.append(new_y)
										else:
											temp_point.append(y_val)
										new_corner_clusters.append(temp_point)
								for i in corner_cluster_centers:
									cv2.circle(temp_img_1, center=(int(i[1]), int(i[0])), radius=2, thickness=-1,
											   color=(255, 0, 255))
								
								cv2.imshow("image", corners)
								cv2.imshow("Cluster", temp_img_1)
								# test = np.copy(corners)
								# test += np.abs(test.min())
								# test *= (255.0 / test.max())
								# cv2.imwrite("Corners_for_report.png", test)
								cv2.imshow('New Corners', new_corners)
		
		cv2.destroyAllWindows()
	
	return


def euclidean_distance(p0, p1):
	"""Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
	
	return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_corners_list(image):
	"""Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
	if image is not None:
		height = 0
		width = 0
		if len(image.shape) == 3:
			height, width, _ = image.shape
		elif len(image.shape) == 2:
			height, width, = image.shape
		top_left = (0, 0)
		top_right = (width - 1, 0)
		bottom_left = (0, height - 1)
		bottom_right = (width - 1, height - 1)
	
		return [top_left, bottom_left, top_right, bottom_right]


def find_markers(image, template=None):
	"""Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
	original_image = np.copy(image)
	is_color = False
	is_simplistic_image = False
	is_noisy = False
	if len(image.shape) > 2:
		if image.shape[2] == 3:
			is_color = True
			unedited_img = np.copy(image)
			temp_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			hsv_image = cv2.cvtColor(cv2.medianBlur(image, 3), cv2.COLOR_BGR2HSV)
			if image.shape[0] == 200 and image.shape[1] == 500:
				is_simplistic_image = True
			
			if is_simplistic_image:
				vals = run_simplistic(image)
				return vals
			pixel_threshold = 0.05
			percentage_of_saturation = np.sum(cv2.calcHist([hsv_image], [1], None,
														   [256], [0, 256])[int(pixel_threshold * 255):-1]) / np.prod(
				hsv_image.shape[0:2])
			if not is_simplistic_image and percentage_of_saturation < 0.95:
				is_noisy = True
	image_noise_removed = cv2.medianBlur(image, ksize=3)
	gray_noise_removed = cv2.cvtColor(image_noise_removed, cv2.COLOR_BGR2GRAY)
	if is_noisy:
		circles, circle_mask = find_circles(grayscale=gray_noise_removed, is_noisy=is_noisy)
	else:
		circles, circle_mask = find_circles(grayscale=gray_noise_removed,
											is_noisy=is_noisy)
	# del pixel_threshold, percentage_of_saturation
	current_image = np.copy(image)
	current_image_grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
	current_image_blur = np.copy(cv2.medianBlur(image, ksize=5))
	current_image_blur_grayscale = cv2.cvtColor(current_image_blur, cv2.COLOR_BGR2GRAY)

	# # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# image_noise_removed = cv2.medianBlur(image, ksize=3)
	# image_morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	# if len(image_noise_removed.shape) > 2:
	# 	grayscale_image_noise_removed = cv2.cvtColor(image_noise_removed, cv2.COLOR_BGR2GRAY)
	# 	grayscale_image_morphed = cv2.cvtColor(image_morphed, cv2.COLOR_BGR2GRAY)
	# 	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# else:
	# 	grayscale_image_noise_removed = image_noise_removed
	# 	grayscale_image_morphed = image_morphed
	# 	grayscale_image = image
	if not is_noisy:
		pixel_val_count = np.unique(current_image_grayscale.ravel(), return_counts=True)
		if (pixel_val_count[1][pixel_val_count[0] == 255] / current_image_grayscale.size) < 0.5:
			is_noisy = True
	if is_noisy:
		corner_mask = find_corners(current_image_blur_grayscale.astype(np.float32), is_noisy=is_noisy,
							   is_simplistic=False)
	else:
		temp_grayscale_image = np.float32(current_image_grayscale)
		corner_mask = find_corners(temp_grayscale_image, is_noisy=is_noisy, is_simplistic=False)
	if corner_mask is not None:
		filtered_points = np.where(corner_mask > 1)
		corner_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
		corner_cluster_centers = None
		if is_noisy or is_color:
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
			corner_clusters = cv2.kmeans(np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T, 4,
										 None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
			if corner_clusters is not None:
				if len(corner_clusters) > 1:
					if corner_clusters[2] is not None:
						if len(corner_clusters[2]) > 0:
							corner_cluster_centers = np.round(corner_clusters[2])
							temp_clusters = np.ravel(corner_cluster_centers)
							cluster_result = np.unique(temp_clusters, return_counts=True)
							new_corner_clusters = []
							for cluster_idx in range(len(corner_cluster_centers)):
								temp_point = []
								x_val = corner_cluster_centers[cluster_idx][0]
								y_val = corner_cluster_centers[cluster_idx][1]
								x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
								y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
								if x_val_count == 1:
									diff = np.abs(corner_cluster_centers[:, 0] - x_val)
									temp_diff = diff == diff[diff > 0].min()
									alt_val = corner_cluster_centers[temp_diff, :][0]
									new_x = (alt_val[0] + x_val) // 2
									temp_point.append(new_x)
								else:
									temp_point.append(x_val)
								
								if y_val_count == 1:
									diff = np.abs(corner_cluster_centers[:, 1] - y_val)
									temp_diff = diff == diff[diff > 0].min()
									alt_val = corner_cluster_centers[temp_diff, :][0]
									new_y = (alt_val[1] + y_val) // 2
									temp_point.append(new_y)
								else:
									temp_point.append(y_val)
								new_corner_clusters.append(temp_point)
	# del new_x, new_y, x_val, y_val, x_val_count, y_val_count, diff, temp_diff, alt_val, temp_point, cluster_idx
	# del corner_clusters, filtered_points
	if not is_noisy:
		ret, thresh = cv2.threshold(current_image_blur_grayscale, 100, 255, 0)
		contours, hierarchy_1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = [contour for contour in contours if 1000 > cv2.contourArea(contour) > 100]
		temp_img = np.zeros(shape=current_image_blur_grayscale.shape)
		cv2.drawContours(temp_img, contours, -1, 255, -1, maxLevel=1)
		contour_points = np.where(temp_img > 1)
		contour_points = np.float32(np.vstack((contour_points[0], contour_points[1]))).T
		contour_cluster_centers = None
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
		contour_clusters = cv2.kmeans(contour_points, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		
		if contour_clusters is not None and is_color:
			if len(contour_clusters) > 1:
				if contour_clusters[2] is not None:
					if len(contour_clusters[2]) > 0:
						contour_cluster_centers = np.round(contour_clusters[2])
						temp_clusters = np.ravel(contour_cluster_centers)
						cluster_result = np.unique(temp_clusters, return_counts=True)
						new_contour_clusters = []
						for cluster_idx in range(len(contour_cluster_centers)):
							temp_point = []
							x_val = contour_cluster_centers[cluster_idx][0]
							y_val = contour_cluster_centers[cluster_idx][1]
							x_val_count = cluster_result[1][cluster_result[0] == x_val][0]
							y_val_count = cluster_result[1][cluster_result[0] == y_val][0]
							if x_val_count == 1:
								diff = np.abs(contour_cluster_centers[:, 0] - x_val)
								temp_diff = diff == diff[diff > 0].min()
								alt_val = contour_cluster_centers[temp_diff, :][0]
								new_x = (alt_val[0] + x_val) // 2
								temp_point.append(new_x)
							else:
								temp_point.append(x_val)
							
							if y_val_count == 1:
								diff = np.abs(contour_cluster_centers[:, 1] - y_val)
								temp_diff = diff == diff[diff > 0].min()
								alt_val = contour_cluster_centers[temp_diff, :][0]
								new_y = (alt_val[1] + y_val) // 2
								temp_point.append(new_y)
							else:
								temp_point.append(y_val)
							new_contour_clusters.append(temp_point)

	if template is None:
		top_left = (0, 0)
		bottom_left = (0, 0)
		top_right = (0, 0)
		bottom_right = (0, 0)
		if not is_noisy and not is_color:
			if len(corner_points) <= 1:
				contour_cluster_centers_unedited = np.copy(contour_cluster_centers)
				top_left = (contour_cluster_centers[:, 1][
									contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])][0],
							np.nanmin(contour_cluster_centers[:, 0]))
				contour_cluster_centers[
					contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				top_right = (contour_cluster_centers[:, 1][
									 contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])][0],
							 np.nanmin(contour_cluster_centers[:, 0]))
				contour_cluster_centers[
					contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				bottom_right = (np.nanmax(contour_cluster_centers[:, 1]), contour_cluster_centers[:, 0][
																					   contour_cluster_centers[:,
																					   1] == np.nanmax(
																						   contour_cluster_centers[:,
																						   1])][0])
				contour_cluster_centers[
					contour_cluster_centers[:, 1] == np.nanmax(contour_cluster_centers[:, 1])] = np.asarray(
					[np.nan, np.nan])
				bottom_left = (np.nanmax(contour_cluster_centers[:, 1]), contour_cluster_centers[:, 0][
																					  contour_cluster_centers[:,
																					  1] == np.nanmax(
																						  contour_cluster_centers[:,
																						  1])][0])
				return [top_left, bottom_left, top_right, bottom_right]
			else:
				return [top_left, bottom_left, top_right, bottom_right]
		
		if corner_cluster_centers is not None and contour_cluster_centers is None:
			if is_noisy:
				corner_cluster_centers_unedited = np.copy(corner_cluster_centers)
				top_left = (corner_cluster_centers[:, 1][
									corner_cluster_centers[:, 0] == np.nanmin(corner_cluster_centers[:, 0])][0],
							np.nanmin(corner_cluster_centers[:, 0]))
				corner_cluster_centers[
					corner_cluster_centers[:, 0] == np.nanmin(corner_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				top_right = (corner_cluster_centers[:, 1][
									 corner_cluster_centers[:, 0] == np.nanmin(corner_cluster_centers[:, 0])][0],
							 np.nanmin(corner_cluster_centers[:, 0]))
				corner_cluster_centers[
					corner_cluster_centers[:, 0] == np.nanmin(corner_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				bottom_right = (np.nanmax(corner_cluster_centers[:, 1]), corner_cluster_centers[:, 0][
																					  corner_cluster_centers[:,
																					  1] == np.nanmax(
																						  corner_cluster_centers[:,
																						  1])][0])
				corner_cluster_centers[
					corner_cluster_centers[:, 1] == np.nanmax(corner_cluster_centers[:, 1])] = np.asarray(
					[np.nan, np.nan])
				bottom_left = (np.nanmax(corner_cluster_centers[:, 1]), corner_cluster_centers[:, 0][
																					 corner_cluster_centers[:,
																					 1] == np.nanmax(
																						 corner_cluster_centers[:, 1])][
																					 0])
				return [top_left, bottom_left, top_right, bottom_right]
			else:
				corner_points_unedited = np.copy(corner_points)
				top_left = (corner_points[:, 1][corner_points[:, 0] == np.nanmin(corner_points[:, 0])][0],
							np.nanmin(corner_points[:, 0]))
				corner_points[corner_points[:, 0] == np.nanmin(corner_points[:, 0])] = np.asarray([np.nan, np.nan])
				top_right = (corner_points[:, 1][corner_points[:, 0] == np.nanmin(corner_points[:, 0])][0],
							 np.nanmin(corner_points[:, 0]))
				corner_points[corner_points[:, 0] == np.nanmin(corner_points[:, 0])] = np.asarray([np.nan, np.nan])
				bottom_right = (np.nanmax(corner_points[:, 1]),
								corner_points[:, 0][corner_points[:, 1] == np.nanmax(corner_points[:, 1])][0])
				corner_points[corner_points[:, 1] == np.nanmax(corner_points[:, 1])] = np.asarray([np.nan, np.nan])
				bottom_left = (np.nanmax(corner_points[:, 1]),
							   (corner_points[:, 0][corner_points[:, 1] == np.nanmax(corner_points[:, 1])][0]))
				return [top_left, bottom_left, top_right, bottom_right]
		elif contour_cluster_centers is not None and corner_cluster_centers is None:
			if is_noisy:
				contour_cluster_centers_unedited = np.copy(contour_cluster_centers)
				top_left = (contour_cluster_centers[:, 1][
									contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])][0],
							np.nanmin(contour_cluster_centers[:, 0]))
				contour_cluster_centers[
					contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				top_right = (contour_cluster_centers[:, 1][
									 contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])][0],
							 np.nanmin(contour_cluster_centers[:, 0]))
				contour_cluster_centers[
					contour_cluster_centers[:, 0] == np.nanmin(contour_cluster_centers[:, 0])] = np.asarray(
					[np.nan, np.nan])
				bottom_right = (np.nanmax(contour_cluster_centers[:, 1]), contour_cluster_centers[:, 0][
																					   contour_cluster_centers[:,
																					   1] == np.nanmax(
																						   contour_cluster_centers[:,
																						   1])][0])
				contour_cluster_centers[
					contour_cluster_centers[:, 1] == np.nanmax(contour_cluster_centers[:, 1])] = np.asarray(
					[np.nan, np.nan])
				bottom_left = (np.nanmax(contour_cluster_centers[:, 1]), contour_cluster_centers[:, 0][
																					  contour_cluster_centers[:,
																					  1] == np.nanmax(
																						  contour_cluster_centers[:,
																						  1])][0])
				return [top_left, bottom_left, top_right, bottom_right]
			else:
				contour_points_unedited = np.copy(contour_points)
				top_left = (contour_points[:, 1][contour_points[:, 0] == np.nanmin(contour_points[:, 0])][0],
							np.nanmin(contour_points[:, 0]))
				contour_points[contour_points[:, 0] == np.nanmin(contour_points[:, 0])] = np.asarray([np.nan, np.nan])
				top_right = (contour_points[:, 1][contour_points[:, 0] == np.nanmin(contour_points[:, 0])][0],
							 np.nanmin(contour_points[:, 0]))
				contour_points[contour_points[:, 0] == np.nanmin(contour_points[:, 0])] = np.asarray([np.nan, np.nan])
				bottom_right = (np.nanmax(contour_points[:, 1]),
								contour_points[:, 0][contour_points[:, 1] == np.nanmax(contour_points[:, 1])][0])
				contour_points[contour_points[:, 1] == np.nanmax(contour_points[:, 1])] = np.asarray([np.nan, np.nan])
				bottom_left = (np.nanmax(contour_points[:, 1]),
							   contour_points[:, 0][contour_points[:, 1] == np.nanmax(contour_points[:, 1])][0])
				return [top_left, bottom_left, top_right, bottom_right]
		elif contour_cluster_centers is not None and corner_cluster_centers is not None:
			if is_color:
				if circles is not None:
					circles_unedited = np.copy(circles[0])
					circles = circles[0]
					if len(circles) > 4:
						circles = circles[:4]
					
					temp_circle_results = find_extrema(np.copy(circles))
					
					temp_val = np.copy(circles[circles[:, 0] == np.nanmin(circles[:, 0])][0])
					idx = np.where(circles == temp_val)
					temp_count = np.unique(idx[0], return_counts=True)
					temp_idx = temp_count[0][np.argmax(temp_count[1])]
					circles[temp_idx] = np.asarray([np.nan, np.nan, np.nan])
					next_val = np.copy(circles[np.nanargmin(circles[:, 0] - temp_val[0])])
					idx = np.where(circles == next_val)
					circles[idx[0][0]] = np.asarray([np.nan, np.nan, np.nan])
					temp_val = np.copy(circles[circles[:, 1] == np.nanmin(circles[:, 1])][0])
					idx = np.where(circles == temp_val)
					temp_count = np.unique(idx[0], return_counts=True)
					temp_idx = temp_count[0][np.argmax(temp_count[1])]
					circles[temp_idx] = np.asarray([np.nan, np.nan, np.nan])
					next_val = np.copy(circles[np.nanargmin(circles[:, 1] - temp_val[1])])
					idx = np.where(circles == next_val)
					circles[idx[0][0]] = np.asarray([np.nan, np.nan, np.nan])
					temp_contour_result = find_extrema(np.copy(contour_cluster_centers))
					temp_corner_results = find_extrema(np.copy(corner_cluster_centers))
					temp_top_left = filter_points(np.asarray([temp_circle_results[0], temp_corner_results[0],
															  temp_contour_result[0]]),
												  is_simplistic=is_simplistic_image)
					temp_bottom_left = filter_points(np.asarray([temp_circle_results[1], temp_corner_results[1],
																 temp_contour_result[1]]),
													 is_simplistic=is_simplistic_image)
					temp_top_right = filter_points(np.asarray([temp_circle_results[2], temp_corner_results[2],
															   temp_contour_result[2]]),
												   is_simplistic=is_simplistic_image)
					temp_bottom_right = filter_points(np.asarray([temp_circle_results[3], temp_corner_results[3],
																  temp_contour_result[3]]),
													  is_simplistic=is_simplistic_image)
					return [temp_top_left, temp_bottom_left, temp_top_right, temp_bottom_right]
		return [top_left, bottom_left, top_right, bottom_right]
	else:
		t = ndimage.rotate(template, 15)
		rgb_image_for_template_matching = cv2.bilateralFilter(cv2.medianBlur(np.copy(original_image), 5), 3, 75, 75)
		grayscale_image_for_template_matching = cv2.cvtColor(rgb_image_for_template_matching, cv2.COLOR_BGR2GRAY)
		template = cv2.bilateralFilter(cv2.medianBlur(np.copy(template), 5), 3, 75, 75)
		template_grayscale = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template_circle_center = (template.shape[1] // 2, template.shape[0] // 2)
		black_extrema_point = np.argwhere(template_grayscale == 0)
		point = black_extrema_point[black_extrema_point[:, 0] == black_extrema_point[:, 0].min()][0]
		radius = template_circle_center[0] - point[0]
		new_template = template[template_circle_center[0] - (radius // 2):template_circle_center[0] + (radius // 2),
					   template_circle_center[1] - (radius // 2):template_circle_center[1] + (radius // 2)]
		new_template_grayscale = cv2.cvtColor(new_template, cv2.COLOR_BGR2GRAY)
		if not is_noisy:
			black_mask = create_black_hsv_mask(hsv_img=cv2.cvtColor(rgb_image_for_template_matching, cv2.COLOR_BGR2HSV))
			white_mask = create_white_hsv_mask(hsv_img=cv2.cvtColor(rgb_image_for_template_matching, cv2.COLOR_BGR2HSV))
			
			filtage = cv2.bitwise_and(rgb_image_for_template_matching, rgb_image_for_template_matching, mask=black_mask)
			contours, hierarchy_1 = cv2.findContours(cv2.cvtColor(filtage, cv2.COLOR_BGR2GRAY),
													 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			test = np.zeros(shape=filtage.shape[:2])
			if contours is not None:
				contours_filtered = [contour for contour in contours if cv2.contourArea(contour) > 300]
				
				cv2.drawContours(test, contours_filtered, -1, 255, -1)
			
			filtered_points = np.where(test > 1)
			
			point_mask = np.zeros(shape=rgb_image_for_template_matching.shape[:2])
			smaller_point_mask = np.zeros(shape=rgb_image_for_template_matching.shape[:2])
			filter_size_threshold = 40
			filter_reduction = 5
			
			if len(filtered_points[0]) > 1:
				filtage_match_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
				filtage_clusters = cv2.kmeans(filtage_match_points, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
				
				if filtage_clusters is not None:
					if len(filtage_clusters) > 2:
						if len(filtage_clusters[2]) > 1:
							points = filtage_clusters[2].astype(np.int)
							for i in points:
								point_mask[i[0] - filter_size_threshold:i[0] + filter_size_threshold,
								i[1] - filter_size_threshold:i[1] + filter_size_threshold] = 255
								smaller_point_mask[i[0] - (filter_size_threshold - filter_reduction):i[0] + (
											filter_size_threshold - filter_reduction),
								i[1] - (filter_size_threshold - filter_reduction):i[1] + (
											filter_size_threshold - filter_reduction)] = 255
		# lines, accumulator = find_lines(tester, np.copy(tester), testing=False)
		# accumulator[smaller_point_mask == 0] = 0
		#
		# # BELOW THIS DO NOT TOUCH -- ALL VERIFIED WORKING
		#
		# # filter needed to smooth outliers
		# accumulator = cv2.filter2D(accumulator, -1, np.ones((3, 3), np.float32) / 9)
		# h, w = accumulator.shape[:2]
		# h = h // 2
		# w = w // 2
		#
		# segment_image_top_left = accumulator[0:w, 0:h]
		# top_left_max = np.argwhere(segment_image_top_left == segment_image_top_left.max())[0]
		#
		# segment_image_top_right = accumulator[0:h, w:]
		# top_right_max = np.argwhere(segment_image_top_right == segment_image_top_right.max())[0]
		# top_right_max[1] += w
		#
		# segment_image_bottom_left = accumulator[h:, 0:w]
		# bottom_left_max = np.argwhere(segment_image_bottom_left == segment_image_bottom_left.max())[0]
		# bottom_left_max[0] += h
		#
		# segment_image_bottom_right = accumulator[h:, w:]
		# bottom_right_max = np.argwhere(segment_image_bottom_right == segment_image_bottom_right.max())[0]
		# bottom_right_max[0] += h
		# bottom_right_max[1] += w
		# # ABOVE THIS DO NOT TOUCH -- ALL VERIFIED WORKING
		
		zeros = np.zeros(shape=rgb_image_for_template_matching.shape[:2])
		
		# https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
		match_method = cv2.TM_SQDIFF
		match_result = cv2.matchTemplate(grayscale_image_for_template_matching.astype(np.float32),
										 new_template_grayscale.astype(np.float32), match_method)
		cv2.normalize(match_result, match_result, 0, 1, cv2.NORM_MINMAX, -1)
		_minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(match_result, None)
		if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
			matchLoc = minLoc
		else:
			matchLoc = maxLoc

		for _ in range(4):
			match_result = cv2.matchTemplate(grayscale_image_for_template_matching.astype(np.float32),
											 new_template_grayscale.astype(np.float32), match_method)
			cv2.normalize(match_result, match_result, 0, 1, cv2.NORM_MINMAX, -1)
			_minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(match_result, None)
			if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
				matchLoc = minLoc
			else:
				matchLoc = maxLoc
			
			result = cv2.rectangle(img=zeros,
								   pt1=matchLoc,
								   pt2=(matchLoc[0] + 10, matchLoc[1] + 10),
								   color=(255), thickness=-1)
			cv2.rectangle(img=grayscale_image_for_template_matching,
						  pt1=matchLoc,
						  pt2=(matchLoc[0] + 10, matchLoc[1] + 10),
						  color=(0), thickness=-1)
			
		filtered_points = np.where(zeros > 1)
		template_match_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
		template_clusters = cv2.kmeans(template_match_points, 4, None, criteria, 10,
									   cv2.KMEANS_RANDOM_CENTERS)
		pts = np.ceil(template_clusters[2])
		new_pts = find_extrema(pts=pts, swap_vals=True)
		loc = np.where(match_result >= 0.9)
		if loc is not None:
			if len(loc) > 0:
				if len(loc[0]) > 10:
					if circles is not None:
						# for pt in zip(*loc[::-1]):
							# cv2.rectangle(zeros, pt, (pt[0] + new_template.shape[0], pt[1] + new_template.shape[1]),
							# 			  255,
							# 			  -1)
						filtered_points = np.where(circle_mask > 1)
						if len(filtered_points[0]) > 1:
							template_match_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
							criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
							template_clusters = cv2.kmeans(template_match_points, 4, None, criteria, 10,
														   cv2.KMEANS_RANDOM_CENTERS)
							if template_clusters is not None:
								if len(template_clusters) > 1:
									if len(template_clusters[2]) > 1:
										template_clusters = template_clusters[2].astype(np.int)
							centers_using_template = template_clusters
							temp_results = find_extrema(centers_using_template, is_simplistic=False, swap_vals=True)
							# for i in temp_results:
							# 	cv2.circle(rgb_image_for_template_matching, center=(i[0], i[1]), color=(0, 255, 0),
							# 			   thickness=-1, radius=2)
							#
							# Could add check to compare circle centers to these results
							# result = check_results(temp_results, circles[:, [0, 1]].astype(np.int),
							# 					   is_noisy=is_noisy)
							temp_results[1] = temp_results[1][0] + 1, temp_results[1][1] + 1

							return [(int(i[0]), int(i[1])) for i in new_pts]
				else:
					# for pt in zip(*loc[::-1]):
					# 	cv2.rectangle(zeros, pt, (pt[0] + new_template.shape[0], pt[1] + new_template.shape[1]), 255,
					# 				  -1)
					template_clusters = None
					filtered_points = np.where(zeros > 1)
					if len(filtered_points[0]) > 1:
						template_match_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
						criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
						template_clusters = cv2.kmeans(template_match_points, 4, None, criteria, 10,
													   cv2.KMEANS_RANDOM_CENTERS)
					if template_clusters is not None:
						if len(template_clusters) > 1:
							if len(template_clusters[2]) > 1:
								template_clusters = template_clusters[2].astype(np.int)
					centers_using_template = template_clusters
					temp_results = find_extrema(centers_using_template, is_simplistic=False, swap_vals=True)
					
					# result = check_results(temp_results, circles[:, [0, 1]].astype(np.int), is_noisy=is_noisy)
					# Could add check to compare circle centers to these results
					return [(int(i[0]), int(i[1])) for i in new_pts]
		
		for pt in zip(*loc[::-1]):
			cv2.rectangle(zeros, pt, (pt[0] + new_template.shape[0], pt[1] + new_template.shape[1]), 255,
						  -1)
		
		template_clusters = None
		filtered_points = np.where(zeros > 1)
		if len(filtered_points[0]) > 1:
			template_match_points = np.float32(np.vstack((filtered_points[0], filtered_points[1]))).T
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
			template_clusters = cv2.kmeans(template_match_points, 4, None, criteria, 10,
										   cv2.KMEANS_RANDOM_CENTERS)
		if template_clusters is not None:
			if len(template_clusters) > 1:
				if len(template_clusters[2]) > 1:
					template_clusters = template_clusters[2].astype(np.int)
		centers_using_template = template_clusters
		temp_results = find_extrema(centers_using_template, is_simplistic=False, swap_vals=True)
		# if circles is not None:
		# 	if len(circles) > 0:
		# 		result = check_results(temp_results, circles[:, [0, 1]].astype(np.int), is_noisy=is_noisy)
		# Could add check to compare circle centers to these results
		return [(int(i[0]), int(i[1])) for i in new_pts]

		
def draw_box(image, markers, thickness=1):
	"""Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
	temp_image = np.copy(image)
	cv2.line(temp_image, markers[0], markers[2], color=(0, 255, 0), thickness=thickness)
	cv2.line(temp_image, markers[1], markers[3], color=(0, 255, 0), thickness=thickness)
	cv2.line(temp_image, markers[0], markers[1], color=(0, 255, 0), thickness=thickness)
	cv2.line(temp_image, markers[2], markers[3], color=(0, 255, 0), thickness=thickness)
	return temp_image


def project_imageA_onto_imageB(imageA, imageB, homography):
	"""Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
	# https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function
	temp_image_a = np.copy(imageA)
	temp_image_b = np.copy(imageB)
	
	homography_inverse = np.linalg.inv(homography)
	
	height, width = temp_image_b.shape[:2]
	indy, indx = np.indices((height, width), dtype=np.float32)
	lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
	
	map_ind = homography_inverse.dot(lin_homg_ind)
	map_x, map_y = map_ind[:-1] / map_ind[-1]
	map_x, map_y = map_x.reshape(height, width).astype(np.float32), map_y.reshape(height, width).astype(np.float32)
	cv2.remap(temp_image_a, map_x, map_y, cv2.INTER_LINEAR, dst=temp_image_b, borderMode=cv2.BORDER_TRANSPARENT)
	return temp_image_b


def find_four_point_transform(src_points, dst_points):
	"""Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
	# https://www.reddit.com/r/computervision/comments/2h1yfj/how_to_calculate_homography_matrix_with_dlt_and/
	container = []
	for (x_src, y_src), (x_dst, y_dst) in zip(src_points, dst_points):
		container.append([x_src, y_src, 1, 0, 0, 0, -x_src * x_dst, -y_src * x_dst, -x_dst])
		container.append([0, 0, 0, x_src, y_src, 1, -x_src * y_dst, -y_src * y_dst, -y_dst])
	A = np.asarray(container)
	u, s, v = np.linalg.svd(A)
	temp = v[-1, :] / v[-1, -1]
	homography_matrix = temp.reshape(3, 3)
	return homography_matrix


def video_frame_generator(filename):
	"""A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
	# Todo: Open file with VideoCapture and set result to 'video'. Replace None
	video = cv2.VideoCapture(filename)
	
	# Do not edit this while loop
	while video.isOpened():
		ret, frame = video.read()
		
		if ret:
			yield frame
		else:
			break
	
	# Todo: Close video (release) and yield a 'None' value. (add 2 lines)
	video.release()
	yield None


def find_aruco_markers(image, aruco_dict=cv2.aruco.DICT_5X5_50):
	"""Finds all ArUco markers and their ID in a given image.

    Hint: you are free to use cv2.aruco module

    Args:
        image (numpy.array): image array.
        aruco_dict (integer): pre-defined ArUco marker dictionary enum.

        For aruco_dict, use cv2.aruco.DICT_5X5_50 for this assignment.
        To find the IDs of markers, use an appropriate function in cv2.aruco module.

    Returns:
        numpy.array: corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        List: list of detected ArUco marker IDs.
    """
	# default_ids = [10, 20, 30, 40]
	# default_value = [[(0, 0) for j in range(4)] for i in default_ids]
	if image is None or len(image.shape) < 2:
		return None, None
	else:
		dictionary = cv2.aruco.Dictionary_get(aruco_dict)
		parameters = cv2.aruco.DetectorParameters_create()
		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image=image, dictionary=dictionary,
																			   parameters=parameters)
	
		if markerCorners is None or len(markerCorners) == 0 or markerIds is None or len(markerIds) == 0:
			return None, None
		else:
			markerIds = np.asarray(markerIds).ravel().tolist()
			dict_of_cords = {markerIds[i]: find_extrema(pts=markerCorners[i][0], swap_vals=False) for i in range(len(markerIds))}
			markerIds = sorted(markerIds)
			return [dict_of_cords[key] for key in markerIds], markerIds


def find_aruco_center(markers):
	"""Draw a bounding box of each marker in image. Also, put a marker ID
        on the top-left of each marker.

    Args:
        image (numpy.array): image array.
        markers (numpy.array): corner coordinate of detected ArUco marker
            in (X, 4, 2) dimension when X is number of detected markers
            and (4, 2) is each corner's x,y coordinate in the order of
            top-left, bottom-left, top-right, and bottom-right.
        ids (list): list of detected ArUco marker IDs.

    Returns:
        List: list of centers of ArUco markers. Each element needs to be
            (x, y) coordinate tuple.
    """
	centers = []
	if markers is None or len(markers) == 0 or markers is None or len(markers) == 0:
		return None
	else:
		for i in markers:
			min_x = min(i[0][0], i[1][0], i[2][0], i[3][0])
			max_x = max(i[0][0], i[1][0], i[2][0], i[3][0])
			min_y = min(i[0][1], i[1][1], i[2][1], i[3][1])
			max_y = max(i[0][1], i[1][1], i[2][1], i[3][1])
			total_x_travel = np.abs(min(i[0][0], i[1][0], i[2][0], i[3][0]) - max(i[0][0], i[1][0], i[2][0], i[3][0]))
			total_y_travel = np.abs(min(i[0][1], i[1][1], i[2][1], i[3][1]) - max(i[0][1], i[1][1], i[2][1], i[3][1]))
			temp_ext = find_extrema(i)
			center = int(min_x + total_x_travel/2), int(min_y + total_y_travel/2)
			centers.append(center)
		return centers
