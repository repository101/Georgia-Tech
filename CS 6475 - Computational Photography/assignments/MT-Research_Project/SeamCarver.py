import numpy as np
import cv2
import sys
import os
import time

from skimage import transform

TESTING = True
SHOW_IMAGE = False

SEAM_PROCESSING = False


# noinspection PyMethodMayBeStatic
class SeamCarver:
	def __init__(self, name, filepath, reduce=True, scale_reduce=None,
				 vertical=None, horizontal=False, expand=False, scale_expand=None, forward_energy=False, useCV2=False):
		self.up_val = [-1, 0]
		self.lh_val = [-1, -1]
		self.rh_val = [-1, 1]
		self.deviations = 3
		self.energy_number = 0
		self.save_directory = None
		self.reduce = reduce
		self.expand = expand
		self.vertical = vertical
		self.horizontal = horizontal
		self.forward_energy = forward_energy
		self.fileName = name
		self.filePath = filepath
		self.fullFilePath = filepath + name
		self.finished = False
		self.FINAL_SEAM_ROWS = None
		self.FINAL_SEAM_COLUMNS = None
		self.energy_map_generated = False
		self.gradient_magnitude_generated = False
		self.bgr_image_data = None
		self.original_bgr_image = None
		self.image_rows = None
		self.image_columns = None
		self.grayscale_image_data = None
		self.path_graph = None
		self.useCV2 = useCV2
		self.path_graph_copy = None
		self.seam = None
		self.temp_seam = None
		self.seam_average_for_expand = None
		self.energy_map = None
		self.energy_map_image = None
		self.blue_energy_channel = None
		self.green_energy_channel = None
		self.red_energy_channel = None
		self.starting_positions = None
		self.path_average_for_expand = None
		self.scale_reduction = scale_reduce
		self.scale_expand = scale_expand
		self.UseMask = None
		self.load_image()

	def __str__(self):
		return "The file name: {}\n" \
			   "The file path: {}".format(self.fileName, self.filePath)

	def testing_function(self, file_name):
		try:
			if self.bgr_image_data is not None:
				cv2.imwrite("{}\\{}_BGR_Image_Data_{}.jpg".format(self.save_directory, file_name,
																  self.energy_number), np.uint8(self.bgr_image_data))

			# if self.bgr_image_data_with_seams is not None:
			# 	cv2.imwrite("{}\\{}_BGR_Image_Data_with_Seams_{}.jpg".format(self.save_directory,
			# 	                                                             file_name, self.energy_number),
			# 	            np.uint8(self.bgr_image_data_with_seams))
			#
			if self.path_graph is not None:
				if self.path_graph.max() > 255:
					temp_path = self.path_graph.copy()
					temp_path *= 255.0 / self.path_graph.max()
					cv2.imwrite("{}\\{}_Path_Graph_{}.jpg".format(self.save_directory, file_name,
																  self.energy_number), np.uint8(temp_path))
				else:
					cv2.imwrite("{}\\{}_Path_Graph_{}.jpg".format(self.save_directory, file_name,
																  self.energy_number), np.uint8(self.path_graph))

			if self.energy_map is not None:
				if self.energy_map.max() > 255:
					temp_energy_map = self.energy_map.copy()
					temp_energy_map *= 255.0 / self.energy_map.max()
					cv2.imwrite("{}\\{}_Energy_Map_{}.jpg".format(self.save_directory, file_name,
																  self.energy_number), np.uint8(temp_energy_map))
				else:
					cv2.imwrite("{}\\{}_Energy_Map_{}.jpg".format(self.save_directory, file_name,
																  self.energy_number), np.uint8(self.energy_map))

			if self.energy_map_image is not None:
				if self.energy_map_image.max() > 255:
					temp_energy_map_image = self.energy_map_image.copy()
					temp_energy_map_image *= 255.0 / self.energy_map_image.max()
					cv2.imwrite("{}\\{}_Energy_Map_Image_{}.jpg".format(self.save_directory,
																		file_name, self.energy_number),
								np.uint8(temp_energy_map_image))
				else:
					cv2.imwrite("{}\\{}_Energy_Map_Image_{}.jpg".format(self.save_directory,
																		file_name, self.energy_number),
								np.uint8(self.energy_map_image))
			if SHOW_IMAGE:
				temp = self.path_graph
				other_temp = np.absolute(self.path_graph)
				other_temp *= 255.0 / other_temp.max()
				temp *= 255.0 / temp.max()
				cv2.imshow("Path Graph", np.uint8(self.path_graph))
				cv2.imshow("Path Graph Other Temp", np.uint8(other_temp))
				cv2.imshow("Path Graph Temp", np.uint8(temp))
				cv2.waitKey(0)
				cv2.destroyAllWindows()

		except Exception as TestingException:
			print("Exception while attempting to run 'newer_routine'.\n", TestingException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	# def newer_routine(self):
	# 	try:
	# 		# Get Energies
	# 		if TESTING:
	# 			self.testing_function("Prior to modifications")
	# 		self.generate_energy_map()
	# 		# -- self.bgr_image_data
	# 		# -- self.energy_map
	# 		# -- self.energy_map_image
	# 		if TESTING:
	# 			self.testing_function("After Generate Energy Map")
	# 		# Map Paths
	#
	# 		self.get_path_graph()
	# 		# -- self.path_graph
	# 		if TESTING:
	# 			self.testing_function("After Seam-Paths")
	#
	# 		# Get Seams
	# 		self.backtrack(self.path_graph)
	# 		if TESTING:
	# 			self.testing_function("After Back-Tracking")
	#
	# 		# Process Seams Remove/Add
	# 		self.seam_processing()
	# 		if TESTING:
	# 			self.testing_function("After Seam-Processing")
	#
	# 	except Exception as NewerRoutineException:
	# 		print("Exception while attempting to run 'newer_routine'.\n", NewerRoutineException)
	# 		exc_type, exc_obj, exc_tb = sys.exc_info()
	# 		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	# 		print(exc_type, fname, exc_tb.tb_lineno)

	def openCvVersion(self):
		try:
			start_time = time.time()
			self.generate_energy_map()
			end_time = time.time()
			print("Feed_Forward_Function Elapsed (with compilation) = %s" % (end_time - start_time))
			cv2.imwrite("TEST BEFORE OPENCV.jpg", self.bgr_image_data)
			testMask = cv2.imread("{}test_mask.jpg".format(self.filePath), cv2.IMREAD_COLOR)
			temp = np.full((testMask.shape[0], testMask.shape[1]), fill_value=True)
			blue, green, red = cv2.split(testMask)
			tester = np.where(blue < 10)
			temp[tester] = False
			path_array_copy = self.energy_map.copy()
			false_mask = temp[temp == False]
			path_array_copy[~temp] = np.random.randint(-90000, -40000, size=false_mask.shape)
			path_array_copy = cv2.GaussianBlur(path_array_copy, (5, 5), 0)
			self.UseMask = path_array_copy

			cv_start = time.time()
			for seam in range(10):
				blue, green, red = cv2.split(self.bgr_image_data)
				blue = transform.seam_carve(blue, self.energy_map, 'Vertical', num=25)
				green = transform.seam_carve(green, self.energy_map, 'Vertical', num=25)
				red = transform.seam_carve(red, self.energy_map, 'Vertical', num=25)
				self.bgr_image_data = cv2.merge((blue, green, red))
			cv_end = time.time()
			blue *= 255
			green *= 255
			red *= 255
			self.bgr_image_data = cv2.merge((blue, green, red))
			print("CV Version Elapsed (with compilation) = %s" % (cv_end - cv_start))
			cv2.imwrite("blue.jpg", blue)
			cv2.imwrite("Result.jpg", self.bgr_image_data)
			print

		except Exception as CVException:
			print("Exception when running OpenCV version.\n", CVException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def load_image(self):
		try:
			self.set_save_directory()
			image = cv2.imread(self.fullFilePath, cv2.IMREAD_COLOR)
			grayscale_image = cv2.imread(self.fullFilePath, cv2.IMREAD_GRAYSCALE)
			self.bgr_image_data = image
			self.original_bgr_image = image.copy()
			self.grayscale_image_data = grayscale_image

			self.image_rows, self.image_columns = self.grayscale_image_data.shape
			if TESTING:
				cv2.imwrite("{}\\Loaded_image.jpg", self.bgr_image_data)

			if self.reduce and not self.expand:
				self.reduce_operations()
			elif self.useCV2 and not self.reduce and not self.expand:
				self.openCvVersion()
			elif self.expand and not self.reduce:
				self.expand_operations()
			else:
				pass
			if TESTING:
				self.add_seams_to_image(self.bgr_image_data, self.FINAL_SEAM_ROWS, self.FINAL_SEAM_COLUMNS,
										original_shape=self.original_bgr_image.shape)
			print("FINISHED")
		except Exception as load_image_err:
			print("Exception when attempting to load the image. \n", load_image_err)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def feed_forward_function(self):
		# Using Energy Map
		try:
			if TESTING:
				print("Generating Feed Forward Map")

			temp_path_graph = np.zeros(self.energy_map.shape)
			rows, columns = self.energy_map.shape
			for i in range(1, rows):
				for j in range(columns - 1):
					val_1 = temp_path_graph[i - 1, j - 1] + \
							np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1])) - \
							np.absolute((self.energy_map[i - 1, j] - self.energy_map[i, j - 1]))
					val_2 = temp_path_graph[i - 1, j] + \
							np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1]))
					val_3 = temp_path_graph[i - 1, j + 1] + \
							np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1])) + \
							np.absolute((self.energy_map[i - 1, j] - self.energy_map[i, j + 1]))

					temp_path_graph[i, j] = temp_path_graph[i, j] + min(val_1, val_2, val_3)
			self.path_graph = temp_path_graph

			testMask = cv2.imread("test_mask.jpg", cv2.IMREAD_COLOR)
			temp = np.zeros((testMask.shape[0], testMask.shape[1]))
			blue, green, red = cv2.split(testMask)
			temp = temp[blue == 0]

			# Insert the first row back into array
			self.path_graph[0, :] = self.energy_map[0, :]
			if TESTING:
				if self.path_graph.min() > 255:
					temp_path_graph = (self.path_graph * 255.0) / self.path_graph.max()
					cv2.imwrite("{}\\Path_Graph_{}.jpg".format(self.save_directory, self.energy_number),
								temp_path_graph)

				else:
					temp_path_graph = np.uint8((self.path_graph * 255.0) / self.path_graph.max())
					temp_path_graph_bone = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_BONE)
					cv2.imwrite("{}\\Path_Graph_{}_Bone.jpg".format(self.save_directory, self.energy_number),
								temp_path_graph_bone)

		except Exception as PathGraphException:
			print("Exception while executing function 'get_path_graph'.\n", PathGraphException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def feed_forward_routine(self):
		try:
			# At this point we have the image read in and we stored a copy in BGR format to self.
			#   We made a copy in self.original_bgr_image, a grayscale in grayscale_image_date
			#   We have the rows and columns for the image.
			for i in range(int(self.bgr_image_data.shape[1] * self.scale_reduction)):
				self.generate_energy_map()
				feed_forward_start = time.time()
				self.feed_forward_function()
				feed_forward_end = time.time()
				print("Feed_Forward_Function Elapsed (with compilation) = %s" % (feed_forward_end - feed_forward_start))
				cv2.imwrite("After_Feed_1.jpg", np.uint8(self.path_graph))
				temp = self.path_graph.copy()
				temp = np.absolute(temp)
				temp *= 255 / temp.max()
				temp = cv2.applyColorMap(np.uint8(temp), cv2.COLORMAP_PINK)
				cv2.imwrite("After_Feed_2.jpg", temp)
				get_path_graph_start = time.time()
				self.get_path_graph()
				get_path_end = time.time()
				print("Get_Path_Graph Elapsed (with compilation) = %s" % (get_path_end - get_path_graph_start))
				cv2.imwrite("After_Get_Path_1.jpg", np.uint8(self.path_graph))
				cv2.imwrite("After_Get_Path_2.jpg", np.uint8(((self.path_graph * 255) / self.path_graph.max())))
				print()

		except Exception as FeedForwardException:
			print("Exception while running 'feed_forward_routine'.\n", FeedForwardException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def set_save_directory(self):
		try:
			directory = ""
			current_directory = os.getcwd()
			if self.useCV2:
				directory = "{}\\OpenCV_Version_Reduction".format(current_directory)
			if self.vertical and not self.horizontal:
				if self.reduce and not self.expand:
					directory = "{}\\Vertical_Seam_Reduction".format(current_directory)
				elif self.expand and not self.reduce:
					directory = "{}\\Vertical_Seam_Expansion".format(current_directory)
			if self.horizontal and not self.vertical:
				if self.reduce and not self.expand:
					directory = "{}\\Horizontal_Seam_Reduction".format(current_directory)
				elif self.expand and not self.reduce:
					directory = "{}\\Horizontal_Seam_Expansion".format(current_directory)
			if not os.path.exists(directory):
				os.makedirs(directory)
			self.save_directory = directory
			return
		except Exception as SaveDirectoryException:
			print("Exception when attempting to set the save directory.\n", SaveDirectoryException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def reduce_operations(self):
		try:
			if TESTING:
				print("Starting Reduce Operations")
			all_seam_rows = np.asarray([])
			all_seam_columns = np.asarray([])
			starting_points = np.asarray([])
			for i in range(int(self.bgr_image_data.shape[1] * self.scale_reduction)):
				if TESTING:
					print("Seam {} Reduction".format(i))
				self.generate_energy_map(vertical=self.vertical, horizontal=self.horizontal)
				self.get_path_graph()
				testMask = cv2.imread("{}test_mask.jpg".format(self.filePath), cv2.IMREAD_COLOR)
				testMask = cv2.resize(testMask, (self.path_graph.shape[1], self.path_graph.shape[0]))
				temp = np.full((testMask.shape[0], testMask.shape[1]), fill_value=True)
				blue, green, red = cv2.split(testMask)
				tester = np.where(blue < 10)
				temp[tester] = False
				path_array_copy = self.path_graph.copy()
				false_mask = temp[temp == False]
				path_array_copy[~temp] = np.random.randint(-90000, -40000, size=false_mask.shape)
				path_array_copy = cv2.GaussianBlur(path_array_copy, (5, 5), 0)
				self.energy_number += 1

				seam_rows, \
				seam_columns, \
				new_starting_points, \
				path_array_copy, \
				path_array_unedited = self.backtrack(path_array_copy, starting_points)

				if len(all_seam_rows) < 1:
					all_seam_rows = seam_rows
					all_seam_columns = seam_columns
				else:
					all_seam_rows = np.vstack([all_seam_rows, seam_rows])
					all_seam_columns = np.vstack([all_seam_columns, seam_columns])

				starting_points = new_starting_points
				if TESTING or SEAM_PROCESSING:
					path_array_copy, \
					self.bgr_image_data = self.seam_processing(path_array_unedited,
															   self.bgr_image_data,
															   seam_rows,
															   seam_columns,
															   i,
															   vertical=self.vertical,
															   horizontal=self.horizontal,
															   reduce=self.reduce,
															   expand=self.expand)
				else:
					path_array_copy, \
					self.bgr_image_data = self.seam_processing(path_array_unedited,
															   self.bgr_image_data,
															   seam_rows,
															   seam_columns,
															   i,
															   vertical=self.vertical,
															   horizontal=self.horizontal,
															   reduce=self.reduce,
															   expand=self.expand)

			# Needed to store these to use later when reversing steps
			self.FINAL_SEAM_ROWS = all_seam_rows
			self.FINAL_SEAM_COLUMNS = all_seam_columns
			if self.vertical and not self.horizontal:
				vertical_horizontal = "Horizontally"
				if self.scale_reduction is not None and self.scale_expand is None:
					reduce_expand_percent = str(self.scale_reduction)
					reduce_expand = "Reduced"
				else:
					reduce_expand_percent = str(self.scale_expand)
					reduce_expand = "Expanded"
			else:
				vertical_horizontal = "Vertically"
				if self.scale_reduction is not None and self.scale_expand is None:
					reduce_expand_percent = str(self.scale_reduction)
					reduce_expand = "Reduced"
				else:
					reduce_expand_percent = str(self.scale_expand)
					reduce_expand = "Expanded"

			cv2.imwrite("{}\\Final Image {} {} by {} Percent.jpg".format(self.save_directory, vertical_horizontal,
																		 reduce_expand,
																		 reduce_expand_percent), self.bgr_image_data)
			return
		except Exception as ReduceOperationException:
			print("Exception when running 'reduce_operations'. \n", ReduceOperationException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def expand_operations(self):
		try:
			if TESTING:
				print("Starting Expand Operations\n")
			all_seam_rows = np.asarray([])
			all_seam_columns = np.asarray([])
			for i in range(int(self.bgr_image_data.shape[1] * self.scale_expand)):
				if TESTING:
					print("Seam {} Expansion".format(i))

				self.generate_energy_map(vertical=self.vertical, horizontal=self.horizontal)
				self.get_path_graph()
				self.energy_number += 1

				if TESTING and SHOW_IMAGE:
					if self.blue_energy_channel is not None:
						cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
					if self.red_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.red_energy_channel))
					if self.green_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.green_energy_channel))
					if self.bgr_image_data is not None:
						cv2.imshow("Line 190 Image", np.uint8(self.bgr_image_data))
					if self.energy_map_image is not None:
						cv2.imshow("Energy", np.uint8(self.energy_map_image))
					if self.path_graph is not None:
						cv2.imshow("Path", np.uint8(self.path_graph))

					cv2.waitKey(0)
					cv2.destroyAllWindows()

				path_array_copy = self.path_graph.copy()
				starting_points = np.asarray([])

				seam_rows, \
				seam_columns, \
				new_starting_points, \
				path_array_copy, \
				path_array_unedited = self.backtrack(path_array_copy, starting_points)

				if TESTING and SHOW_IMAGE:
					if self.blue_energy_channel is not None:
						cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
					if self.red_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.red_energy_channel))
					if self.green_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.green_energy_channel))
					if self.bgr_image_data is not None:
						cv2.imshow("Line 216 Image", np.uint8(self.bgr_image_data))
					if self.energy_map_image is not None:
						cv2.imshow("Energy", np.uint8(self.energy_map_image))
					if self.path_graph is not None:
						cv2.imshow("Path", np.uint8(self.path_graph))

					cv2.waitKey(0)
					cv2.destroyAllWindows()

				if len(all_seam_rows) < 1:
					all_seam_rows = seam_rows
					all_seam_columns = seam_columns
				else:
					all_seam_rows = np.vstack([all_seam_rows, seam_rows])
					all_seam_columns = np.vstack([all_seam_columns, seam_columns])

				path_array_copy, \
				self.bgr_image_data = self.seam_processing(path_array_unedited,
														   self.bgr_image_data,
														   seam_rows,
														   seam_columns,
														   i,
														   vertical=self.vertical,
														   horizontal=self.horizontal,
														   reduce=self.reduce,
														   expand=self.expand)
				if TESTING and SHOW_IMAGE:
					if self.blue_energy_channel is not None:
						cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
					if self.red_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.red_energy_channel))
					if self.green_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.green_energy_channel))
					if self.bgr_image_data is not None:
						cv2.imshow("Line 252 Image", np.uint8(self.bgr_image_data))
					if self.energy_map_image is not None:
						cv2.imshow("Energy", np.uint8(self.energy_map_image))
					if self.path_graph is not None:
						cv2.imshow("Path", np.uint8(self.path_graph))

					cv2.waitKey(0)
					cv2.destroyAllWindows()
			# Needed to store these to use later when reversing steps
			self.FINAL_SEAM_ROWS = all_seam_rows
			if self.FINAL_SEAM_ROWS is not None:
				np.savetxt("Final Seam Rows.csv", self.FINAL_SEAM_ROWS, delimiter=",", fmt="%1.2f")
			self.FINAL_SEAM_COLUMNS = all_seam_columns
			if self.FINAL_SEAM_COLUMNS is not None:
				np.savetxt("Final Seam Columns.csv", self.FINAL_SEAM_COLUMNS, delimiter=",", fmt="%1.2f")

			vertical_horizontal = "n"
			reduce_expand_percent = "n"
			if TESTING:
				if self.vertical and not self.horizontal:
					vertical_horizontal = "Horizontally"
					if self.scale_reduction is not None and self.scale_expand is None:
						reduce_expand_percent = str(self.scale_reduction)
						reduce_expand = "Reduced"
					else:
						reduce_expand_percent = str(self.scale_expand)
						reduce_expand = "Expanded"
				else:
					vertical_horizontal = "Vertically"
					if self.scale_reduction is not None and self.scale_expand is None:
						reduce_expand_percent = str(self.scale_reduction)
						reduce_expand = "Reduced"
					else:
						reduce_expand_percent = str(self.scale_expand)
						reduce_expand = "Expanded"

				cv2.imwrite("{}\\Final Image {} {} by {} Percent.jpg".format(self.save_directory, vertical_horizontal,
																			 reduce_expand,
																			 reduce_expand_percent),
							self.bgr_image_data)
			else:
				cv2.imwrite(
					"{}\\Final Image Modified by {} Percent.jpg".format(self.save_directory, vertical_horizontal,
																		reduce_expand_percent), self.bgr_image_data)

		except Exception as ExpandOperationsExceptions:
			print("Exception when running 'expand_operations' function.\n", ExpandOperationsExceptions)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def update_deviation(self, new_deviation):
		if self.deviations != new_deviation:
			self.deviations = new_deviation
		return

	def get_path_graph(self, vertical=True, horizontal=False):
		# Using Energy Map
		try:
			if self.forward_energy:
				if TESTING:
					print("Generating Feed Forward Map")
				temp_path_graph = np.zeros(self.energy_map.shape)
				rows, columns = self.energy_map.shape
				for i in range(1, rows):
					for j in range(columns - 1):
						val_1 = temp_path_graph[i - 1, j - 1] + \
								np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1])) - \
								np.absolute((self.energy_map[i - 1, j] - self.energy_map[i, j - 1]))
						val_2 = temp_path_graph[i - 1, j] + \
								np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1]))
						val_3 = temp_path_graph[i - 1, j + 1] + \
								np.absolute((self.energy_map[i, j + 1] - self.energy_map[i, j - 1])) + \
								np.absolute((self.energy_map[i - 1, j] - self.energy_map[i, j + 1]))

						temp_path_graph[i, j] = temp_path_graph[i, j] + min(val_1, val_2, val_3)
				self.path_graph = temp_path_graph

				# Insert the first row back into array
				self.path_graph[0, :] = self.energy_map[0, :]
				if TESTING:
					if self.path_graph.min() > 255:
						temp_path_graph = (self.path_graph * 255.0) / self.path_graph.max()
						cv2.imwrite("{}\\Path_Graph_{}.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph)

					else:
						temp_path_graph = np.uint8((self.path_graph * 255.0) / self.path_graph.max())
						temp_path_graph_bone = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_BONE)
						cv2.imwrite("{}\\Path_Graph_{}_Bone.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph_bone)
			else:
				if TESTING:
					print("Generating Path Map")
				self.starting_positions = np.empty(shape=(0, self.energy_map.shape[1]))
				rows, cols = self.energy_map.shape
				self.path_graph = self.energy_map.copy()
				tes_rows = np.asarray([np.full(shape=(1, self.path_graph.shape[1]), fill_value=i)
									   for i in range(0, self.path_graph.shape[0])])
				tes_rows = np.squeeze(tes_rows)
				tes_rows = np.ravel(tes_rows)
				tp = self.energy_map.copy()
				tp = np.ravel(tp)

				tes_cols = np.asarray([np.arange(cols) for i in range(0, rows)])
				tes_cols = np.squeeze(tes_cols)
				tes_cols = np.ravel(tes_cols)

				# This is not necessarily the vertical graph, its any graph
				vertical_graph = np.vectorize(self.vectorized_dag)
				vertical_graph(tp, tes_rows, tes_cols)

				# Insert the first row back into array
				self.path_graph[0, :] = self.energy_map[0, :]
				if TESTING:
					if self.path_graph.min() > 255:
						temp_path_graph = (self.path_graph * 255.0) / self.path_graph.max()
						cv2.imwrite("{}\\Path_Graph_{}.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph)

					else:
						temp_path_graph = np.uint8((self.path_graph * 255.0) / self.path_graph.max())
						# if horizontal and not vertical:
						# 	temp_path_graph = np.uint8(np.rot90(temp_path_graph, k=-1, axes=(1, 0)))
						temp_path_graph_Bone = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_BONE)
						temp_path_graph_Hot = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_HOT)
						temp_path_graph_Jet = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_JET)
						temp_path_graph_Parula = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_PARULA)
						temp_path_graph_Pink = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_PINK)
						temp_path_graph_Rainbow = cv2.applyColorMap(temp_path_graph, cv2.COLORMAP_RAINBOW)

						cv2.imwrite("{}\\Path_Graph_{}.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph)
						cv2.imwrite(
							"{}\\Path_Graph_{}_Bone.jpg".format(self.save_directory, self.energy_number),
							temp_path_graph_Bone)
						cv2.imwrite("{}\\Path_Graph_{}_Hot.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph_Hot)
						cv2.imwrite("{}\\Path_Graph_{}_Jet.jpg".format(self.save_directory, self.energy_number),
									temp_path_graph_Jet)
						cv2.imwrite(
							"{}\\Path_Graph_{}_Parula.jpg".format(self.save_directory, self.energy_number),
							temp_path_graph_Parula)
						cv2.imwrite(
							"{}\\Path_Graph_{}_Pink.jpg".format(self.save_directory, self.energy_number),
							temp_path_graph_Pink)
						cv2.imwrite(
							"{}\\Path_Graph_{}_Rainbow.jpg".format(self.save_directory, self.energy_number),
							temp_path_graph_Rainbow)

					print("Finished Generating Path Map")
				if horizontal and not vertical:
					# TODO Implement the
					pass

		except Exception as PathGraphException:
			print("Exception while executing function 'get_path_graph'.\n", PathGraphException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def get_average_value(self, current_ix, image=True, path=False):
		try:
			left_val = [0, -1]
			right_val = [0, 1]
			left_value = None
			current_value = None
			right_value = None
			# region Populate the index for the neighboring pixels
			left_idx = np.asarray([current_ix[0], current_ix[1] + left_val[1]])
			right_idx = np.asarray([current_ix[0], current_ix[1] + right_val[1]])
			# endregion

			# region Validate Coordinates of the 3 possible new values
			if image and not path:
				set_image = self.bgr_image_data
			else:
				set_image = self.path_graph

			if not self.validate_coordinates(set_image.shape, left_idx):
				pass
			else:
				left_value = set_image[left_idx[0], left_idx[1]]
			if not self.validate_coordinates(set_image.shape, right_idx):
				pass
			else:
				right_value = set_image[right_idx[0], right_idx[1]]
			if not self.validate_coordinates(set_image.shape, current_ix):
				pass
			else:
				current_value = set_image[current_ix[0], current_ix[1]]
			# endregion\
			# region Only work with the neighbors which validated correctly
			avg_array = []
			if left_value is not None:
				avg_array.append(left_value)
			if right_value is not None:
				avg_array.append(right_value)
			if current_value is not None:
				avg_array.append(current_value)
			# endregion
			avg_array = np.asarray(avg_array)
			if image and not path:
				if len(avg_array) > 0:
					return_val = np.average(avg_array, axis=1)
					if len(return_val) <= 2:
						return_val = np.append(return_val, values=np.average(return_val))
					return return_val
			else:
				return np.asarray(np.average(avg_array))
		except Exception as GetAverageValueException:
			print("Exception while running 'get_average_value' function.\n", GetAverageValueException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def backtrack(self, path_array, previously_visited=None):
		try:
			if TESTING:
				print("Beginning Backtracking")
			up_val = [-1, 0]
			lh_val = [-1, -1]
			rh_val = [-1, 1]

			if TESTING and SHOW_IMAGE:
				if self.blue_energy_channel is not None:
					cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
				if self.red_energy_channel is not None:
					cv2.imshow("Red", np.uint8(self.red_energy_channel))
				if self.green_energy_channel is not None:
					cv2.imshow("Red", np.uint8(self.green_energy_channel))
				if self.bgr_image_data is not None:
					cv2.imshow("Line 419 Image", np.uint8(self.bgr_image_data))
				if self.energy_map_image is not None:
					cv2.imshow("Energy", np.uint8(self.energy_map_image))
				if self.path_graph is not None:
					cv2.imshow("Path", np.uint8(self.path_graph))

				cv2.waitKey(0)
				cv2.destroyAllWindows()

			start_row = path_array.shape[0] - 1
			temp = path_array[-1, :]
			start_col = np.argmin(path_array[-1, :])

			if previously_visited is not None and len(previously_visited) > 0:
				if np.any(previously_visited == start_col):
					path_array_unedited = path_array.copy()
					path_array[start_row, start_col] = 43000.0
					start_col = np.argmin(path_array[-1, :])
					current_ix = np.asarray([start_row, start_col])
					previously_visited = np.append(previously_visited, current_ix[1])
				else:
					path_array_unedited = path_array.copy()
					path_array[start_row, start_col] = 43000.0
					current_ix = np.asarray([start_row, start_col])
					previously_visited = np.append(previously_visited, current_ix[1])

			else:
				path_array_unedited = path_array.copy()
				current_ix = np.asarray([start_row, start_col])
				previously_visited = np.asarray([current_ix[1]])
				path_array[start_row, start_col] = 9 ** 8

			temp_seam = None
			path_average = None
			seam_average_for_expand = None
			next_ix = None
			run_back = True

			while run_back:
				if temp_seam is None:
					temp_seam = current_ix
					temp_avg = self.get_average_value(current_ix, image=True)
					seam_average_for_expand = temp_avg

					temp_path_average = self.get_average_value(current_ix, image=False, path=True)
					path_average = temp_path_average
				else:
					temp_seam = np.vstack([temp_seam, next_ix])
					temp_avg = self.get_average_value(next_ix, image=True)
					seam_average_for_expand = np.vstack([seam_average_for_expand, temp_avg])
					temp_path_average = self.get_average_value(current_ix, image=False, path=True)
					path_average = np.vstack([path_average, temp_path_average])

				if next_ix is not None:
					current_ix = next_ix

				# region Populate the index for the neighboring pixels
				lh_idx = np.asarray([current_ix[0] + lh_val[0], current_ix[1] + lh_val[1]])
				up_idx = np.asarray([current_ix[0] + up_val[0], current_ix[1] + up_val[1]])
				rh_idx = np.asarray([current_ix[0] + rh_val[0], current_ix[1] + rh_val[1]])
				# endregion

				left = None
				up = None
				right = None

				# region Validate Coordinates of the 3 possible new values
				if not self.validate_coordinates(path_array.shape, lh_idx):
					pass
				else:
					left = path_array[lh_idx[0], lh_idx[1]]
				if not self.validate_coordinates(path_array.shape, up_idx):
					pass
				else:
					up = path_array[up_idx[0], up_idx[1]]
				if not self.validate_coordinates(path_array.shape, rh_idx):
					pass
				else:
					right = path_array[rh_idx[0], rh_idx[1]]
				# endregion

				# region Only work with the neighbors which validated correctly
				min_array = []
				if left is not None:
					min_array.append(left)
				if up is not None:
					min_array.append(up)
				if right is not None:
					min_array.append(right)
				# endregion

				# region If we haven't any neighbors to process we exit the loop,
				if len(min_array) == 0:
					break
				# endregion
				else:
					# region
					min_array = np.asarray(min_array)
					min_val = np.min(min_array)
					if min_val == left:
						next_ix = lh_idx
					elif min_val == up:
						next_ix = up_idx
					elif min_val == right:
						next_ix = rh_idx
					else:
						# Another chance to leave the while loop
						break
					# endregion
					unique_vals = np.unique(min_array)
					unique_count = 0

					# region
					if len(unique_vals) <= 2:
						# check if there are two instances of the same value, such as both up and left have the same values
						if unique_vals[0] == left:
							unique_count += 1
						if unique_vals[0] == up:
							unique_count += 1
						if unique_vals[0] == right:
							unique_count += 1
					# endregion

					# region At this point we have a situation where we have two possible choices
					if unique_count > 1:
						# At this point we could the seam can fork, I am not sure how I will decide to handle
						#   possibly look at the next neighbor of each and see which is overall lower
						pass
			# endregion
			# Need to flip seam_average_for_expand b/c we start from the bottom of the array and work back
			#  so the array is upside down
			seam_average_for_expand = np.flipud(seam_average_for_expand)
			path_average = np.flipud(path_average)
			self.seam_average_for_expand = seam_average_for_expand
			self.path_average_for_expand = path_average
			if TESTING:
				print("Finished Backtracking")
			return temp_seam[:, 0], temp_seam[:, 1], previously_visited, path_array, path_array_unedited

		except Exception as BackTrackException:
			print("Exception while executing function 'backtrack'.\n", BackTrackException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def seam_processing(self, path_array, image, seam_rows, seam_cols, img_id,
						vertical=True, horizontal=False, reduce=True, expand=False):
		try:
			if TESTING:
				print("Starting Seam Processing")
				if SHOW_IMAGE:
					if self.blue_energy_channel is not None:
						cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
					if self.red_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.red_energy_channel))
					if self.green_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.green_energy_channel))
					if self.bgr_image_data is not None:
						cv2.imshow("Line 577 Image", np.uint8(self.bgr_image_data))
					if self.energy_map_image is not None:
						cv2.imshow("Energy", np.uint8(self.energy_map_image))
					if self.path_graph is not None:
						cv2.imshow("Path", np.uint8(self.path_graph))

					cv2.waitKey(0)
					cv2.destroyAllWindows()

			if vertical and not horizontal:
				pass
			elif horizontal and not vertical:
				image = np.rot90(image, k=1, axes=(1, 0))
			if reduce and not expand:
				# Create Mask for reduction only
				mask = np.full(shape=path_array.shape, fill_value=True)
				mask[seam_rows, seam_cols] = False
				new_path_array = path_array[mask]
				new_image = image.copy()
				new_image = new_image[mask]
				if TESTING or SEAM_PROCESSING:
					path_array_without_seam = path_array.copy()
					# Scale array
					path_array_without_seam *= 255.0 / path_array_without_seam.max()
					path_with_seam = path_array_without_seam.copy()
					path_with_seam[~mask] = 0

					blue, green, red = cv2.split(image)
					blue = blue[mask]
					green = green[mask]
					red = red[mask]
					new_path_array = new_path_array.reshape(path_array.shape[0], (path_array.shape[1] - 1))
					blue = blue.reshape(image.shape[0], (image.shape[1] - 1))
					green = green.reshape(image.shape[0], (image.shape[1] - 1))
					red = red.reshape(image.shape[0], (image.shape[1] - 1))
					if horizontal and not vertical:
						blue = np.rot90(blue, k=-1, axes=(1, 0))
						green = np.rot90(green, k=-1, axes=(1, 0))
						red = np.rot90(red, k=-1, axes=(1, 0))
					new_image = cv2.merge((blue, green, red))
					if TESTING:
						if horizontal and not vertical:
							mask = np.rot90(mask, k=-1, axes=(1, 0))
							path_array_without_seam = np.rot90(path_array_without_seam, k=-1, axes=(1, 0))
							path_with_seam = np.rot90(path_with_seam, k=-1, axes=(1, 0))
						cv2.imwrite("{}\\Image_{}_with_seams.jpg".format(self.save_directory, img_id), new_image)
						if path_with_seam.min() < 0:
							temp_path_with_seam = np.absolute(path_with_seam)
							temp_path_with_seam *= 255.0 / temp_path_with_seam.max()
							cv2.imwrite("{}\\Path_array_with_seam_{}.jpg".format(self.save_directory, img_id),
										temp_path_with_seam)
						else:
							cv2.imwrite("{}\\Path_array_with_seam_{}.jpg".format(self.save_directory, img_id),
										path_with_seam)
						if path_array_without_seam.min() < 0:
							temp_path_without_seam = np.absolute(path_array_without_seam)
							temp_path_without_seam *= 255.0 / temp_path_without_seam.max()
							cv2.imwrite("{}\\Path_array_without_seam_{}.jpg".format(self.save_directory, img_id),
										temp_path_without_seam)
						else:
							cv2.imwrite("{}\\Path_array_without_seam_{}.jpg".format(self.save_directory, img_id),
										path_array_without_seam)
						cv2.imwrite("{}\\Mask_array_{}.jpg".format(self.save_directory, img_id),
									mask.astype(np.uint8) * 255)

				else:
					new_image = np.reshape(new_image, newshape=(image.shape[0], (image.shape[1] - 1), 3))
					new_path_array = new_path_array.reshape(path_array.shape[0], (path_array.shape[1] - 1))
				if horizontal and not vertical and not TESTING and not SEAM_PROCESSING:
					new_image = np.rot90(new_image, k=-1, axes=(1, 0))
				if TESTING:
					print("Finished Seam Processing")
					if SHOW_IMAGE:
						if self.blue_energy_channel is not None:
							cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
						if self.red_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.red_energy_channel))
						if self.green_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.green_energy_channel))
						if self.bgr_image_data is not None:
							cv2.imshow("Line 637 Image", np.uint8(self.bgr_image_data))
						if self.energy_map_image is not None:
							cv2.imshow("Energy", np.uint8(self.energy_map_image))
						if self.path_graph is not None:
							cv2.imshow("Path", np.uint8(self.path_graph))

						cv2.waitKey(0)
						cv2.destroyAllWindows()
				return new_path_array, new_image
			elif expand and not reduce:
				rows, columns, channels = image.shape
				new_bgr = np.empty(shape=(rows, columns + 1, channels))
				new_path_array = np.empty(shape=(rows, columns + 1))

				for i in range(len(seam_rows) - 1):
					value_to_insert_for_image = self.seam_average_for_expand[seam_rows[i]]

					value_to_insert_for_path = self.path_average_for_expand[seam_rows[i]]
					new_bgr[seam_rows[i]] = np.insert(image[seam_rows[i]], seam_cols[i] + 1,
													  value_to_insert_for_image, axis=0)
					new_path_array[seam_rows[i]] = np.insert(path_array[seam_rows[i]], seam_cols[i] + 1,
															 value_to_insert_for_path, axis=0)
				if horizontal and not vertical:
					new_bgr = np.rot90(new_bgr, k=-1, axes=(1, 0))
				if TESTING:
					print("Starting Seam Processing")
					if SHOW_IMAGE:
						if self.blue_energy_channel is not None:
							cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
						if self.red_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.red_energy_channel))
						if self.green_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.green_energy_channel))
						if self.bgr_image_data is not None:
							cv2.imshow("Line 668 Image", np.uint8(new_bgr))
						if self.energy_map_image is not None:
							cv2.imshow("Energy", np.uint8(self.energy_map_image))
						if self.path_graph is not None:
							cv2.imshow("Path", np.uint8(new_path_array))

						cv2.waitKey(0)
						cv2.destroyAllWindows()
					print("Finished Seam Processing")
				return new_path_array, new_bgr
			else:
				pass
		except Exception as SeamProcessingException:
			print("Exception while running function 'remove_seam_from_path_array'.", SeamProcessingException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def vectorized_dag(self, data_value, rows, col):

		# region Coordinate Generation
		lh_idx = [rows + self.lh_val[0], col + self.lh_val[1]]
		up_idx = [rows + self.up_val[0], col + self.up_val[1]]
		rh_idx = [rows + self.rh_val[0], col + self.rh_val[1]]
		# endregion
		left = None
		up = None
		right = None

		# region Coordinate Validation
		if not self.validate_coordinates(self.path_graph.shape, lh_idx):
			pass
		else:
			left = self.path_graph[lh_idx[0], lh_idx[1]]

		if not self.validate_coordinates(self.path_graph.shape, up_idx):
			pass
		else:
			up = self.path_graph[up_idx[0], up_idx[1]]

		if not self.validate_coordinates(self.path_graph.shape, rh_idx):
			pass
		else:
			right = self.path_graph[rh_idx[0], rh_idx[1]]
		# endregion

		min_array = []
		if left is not None:
			min_array.append(left)
		if up is not None:
			min_array.append(up)
		if right is not None:
			min_array.append(right)
		if len(min_array) == 0:
			pass
		else:
			min_array = np.asarray(min_array)
			min_val = np.min(min_array)
			self.path_graph[rows, col] = data_value + min_val
			return
		return

	def generate_energy_map(self, sobel=True, vertical=True, horizontal=False):
		"""
		Generation of the energy maps for a given image. There are multiple methods for obtaining the energies of
		an image. I am using the sobel filters which have built in gaussian blur, which will allow for less artifacts
		between images. I split the image into its three color channels BGR and process them individually in both the
		x and y directions. For each channel I combine the sobel_x and sobel_y. Once finished processing the 3 channels
		I combine them using cv2.merge to get a combined energy map.
		:param sobel:
		:param vertical:
		:param horizontal:
		:param alternating:
		:return:

		"""
		# Another Source
		# http://cs.brown.edu/courses/cs129/results/proj3/taox/
		try:
			if TESTING:
				print("Generating Energy Map")
			channel_name = ["Blue", "Green", "Red"]
			bgr_image = self.bgr_image_data.copy()
			if vertical:
				pass
			elif horizontal:
				# Rotate image
				bgr_image = np.rot90(bgr_image, k=1, axes=(1, 0))
			else:
				pass
			# This is alternating... different situation, take care of later
			channels = blue, green, red = cv2.split(bgr_image)
			if TESTING:
				if SHOW_IMAGE:
					if self.blue_energy_channel is not None:
						cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
					if self.red_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.red_energy_channel))
					if self.green_energy_channel is not None:
						cv2.imshow("Red", np.uint8(self.green_energy_channel))
					if self.bgr_image_data is not None:
						cv2.imshow("Line 741 Image", np.uint8(self.bgr_image_data))
					if self.energy_map_image is not None:
						cv2.imshow("Energy", np.uint8(self.energy_map_image))
					if self.path_graph is not None:
						cv2.imshow("Path", np.uint8(self.path_graph))

					cv2.waitKey(0)
					cv2.destroyAllWindows()

			if sobel:
				for sub_channel in range(len(channels)):
					sobel_x = cv2.Sobel(channels[sub_channel], cv2.CV_64F, 1, 0, ksize=self.deviations)
					sobel_x_abs = np.absolute(sobel_x)
					sobel_y = cv2.Sobel(channels[sub_channel], cv2.CV_64F, 0, 1, ksize=self.deviations)
					sobel_y_abs = np.absolute(sobel_y)
					sub_channel_energy = np.absolute(np.add(sobel_x, sobel_y))
					if TESTING:
						cv2.imwrite(
							"{}\\sub_channel_{}_{}_sobel_x.jpg".format(self.save_directory, channel_name[sub_channel],
																	   sub_channel), sobel_x)
						cv2.imwrite("{}\\sub_channel_{}_{}_sobel_x_abs.jpg".format(self.save_directory,
																				   channel_name[sub_channel],
																				   sub_channel), sobel_x_abs)
						cv2.imwrite(
							"{}\\sub_channel_{}_{}_sobel_y.jpg".format(self.save_directory, channel_name[sub_channel],
																	   sub_channel), sobel_y)
						cv2.imwrite("{}\\sub_channel_{}_{}_sobel_y_abs.jpg".format(self.save_directory,
																				   channel_name[sub_channel],
																				   sub_channel), sobel_y_abs)
						cv2.imwrite(
							"{}\\sub_channel_{}_{}_energy.jpg".format(self.save_directory, channel_name[sub_channel],
																	  sub_channel), sub_channel_energy)

					if sub_channel == 0:
						self.blue_energy_channel = sub_channel_energy
					elif sub_channel == 1:
						self.green_energy_channel = sub_channel_energy
					elif sub_channel == 2:
						self.red_energy_channel = sub_channel_energy
					else:
						continue

				self.energy_map = self.blue_energy_channel * 0.33 + \
								  self.green_energy_channel * 0.33 + \
								  self.red_energy_channel * 0.33

				self.energy_map_image = cv2.merge((self.blue_energy_channel,
												   self.green_energy_channel,
												   self.red_energy_channel))

				if TESTING:
					name = 'Vertical'
					if vertical:
						pass
					elif horizontal:
						name = "Horizontal"

					cv2.imwrite("{}\\{}_sub_channel_combined_energy_{}.jpg".format(self.save_directory, name,
																				   self.energy_number), self.energy_map)
					cv2.imwrite("{}\\{}_energy_map_image_{}.jpg".format(self.save_directory, name, self.energy_number),
								self.energy_map_image)
					if SHOW_IMAGE:
						if self.blue_energy_channel is not None:
							cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
						if self.red_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.red_energy_channel))
						if self.green_energy_channel is not None:
							cv2.imshow("Red", np.uint8(self.green_energy_channel))
						if self.bgr_image_data is not None:
							cv2.imshow("Line 803 Image", np.uint8(self.bgr_image_data))
						if self.energy_map_image is not None:
							cv2.imshow("Energy", np.uint8(self.energy_map_image))
						if self.path_graph is not None:
							cv2.imshow("Path", np.uint8(self.path_graph))

						cv2.waitKey(0)
						cv2.destroyAllWindows()
			if TESTING:
				print("Finished Generating Energy Map")
			return
		except Exception as GenerateEnergyMapException:
			print("Exception while running function 'generate_energy_map'. \n", GenerateEnergyMapException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def add_seams_to_image(self, image, seam_rows, seam_cols, original_shape=None):
		"""

		:param image:
		:param seam_rows:
		:param seam_cols:
		:param save:
		:param vertical:
		:param horizontal:
		:param original_shape:
		"""
		try:
			if TESTING:
				print("Adding Seams to Image Started")
			if seam_rows is None or seam_cols is None:
				print()
			if original_shape is not None:
				blue, green, red = cv2.split(image)
				for operation in range(len(seam_rows) - 1, -1, -1):
					new_blue_image = np.zeros(shape=(blue.shape[0], blue.shape[1] + 1))
					new_green_image = np.zeros(shape=(green.shape[0], green.shape[1] + 1))
					new_red_image = np.zeros(shape=(red.shape[0], red.shape[1] + 1))

					for row in range(len(seam_rows[operation]) - 1, -1, -1):
						index = seam_cols[operation][row]
						row = seam_rows[operation][row]
						blue_channel = blue[seam_rows[operation][row]]
						green_channel = green[seam_rows[operation][row]]
						red_channel = red[seam_rows[operation][row]]
						temp_blue = self.insert_into_sub_array(blue_channel, 0, index)
						temp_green = self.insert_into_sub_array(green_channel, 0, index)
						temp_red = self.insert_into_sub_array(red_channel, 255, index)
						total_rows = new_blue_image.shape[0] - 1
						new_blue_image[total_rows - row] = temp_blue
						new_green_image[total_rows - row] = temp_green
						new_red_image[total_rows - row] = temp_red

					blue = new_blue_image
					green = new_green_image
					red = new_red_image
					new_image = cv2.merge((blue, green, red))
					if TESTING:
						# cv2.imwrite("{}\\Seam_{}_Blue_Channel.jpg".format(self.save_directory, operation), blue)
						# cv2.imwrite("{}\\Seam_{}_Green_Channel.jpg".format(self.save_directory, operation), green)
						# cv2.imwrite("{}\\Seam_{}_Red_Channel.jpg".format(self.save_directory, operation), red)
						cv2.imwrite("{}\\Seam_{}_Image.jpg".format(self.save_directory, operation), new_image)
						if SHOW_IMAGE:
							if self.blue_energy_channel is not None:
								cv2.imshow("Blue", np.uint8(self.blue_energy_channel))
							if self.red_energy_channel is not None:
								cv2.imshow("Red", np.uint8(self.red_energy_channel))
							if self.green_energy_channel is not None:
								cv2.imshow("Red", np.uint8(self.green_energy_channel))
							if self.bgr_image_data is not None:
								cv2.imshow("ADD SEAMS TO IMAGE Image", np.uint8(self.bgr_image_data))
							if self.energy_map_image is not None:
								cv2.imshow("Energy", np.uint8(self.energy_map_image))
							if self.path_graph is not None:
								cv2.imshow("Path", np.uint8(self.path_graph))

							cv2.waitKey(0)
							cv2.destroyAllWindows()
			if TESTING:
				print("Adding Seams to Images finished")
		except Exception as AddSeamsToImageException:
			print("Exception while attempting to add the seams to the image.", AddSeamsToImageException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def validate_filename(self):
		"""

		:return:
		"""
		try:
			if "." in self.fileName:
				# Then the file "Should" have a file extension
				split_name = self.fileName.split(".")
				if len(split_name[-1]) < 3:
					# Then the file extension is invalid, adding a default extension of .jpg
					self.fileName = split_name[0] + ".jpg"
			elif len(self.fileName) > 2:
				# Filename is holding information but lacks a file extension, so we add one
				self.fileName = self.fileName + ".jpg"

			return True
		except Exception as validate_err:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def insert_into_sub_array(self, sub_array, value, location):
		"""

		:param sub_array:
		:param value:
		:param location:
		:return:
		"""
		try:
			result = np.insert(sub_array, location, value)
			return result
		except Exception as InsertIntoArrayException:
			print("Exception when running 'insert_into_sub_array' function.\n", InsertIntoArrayException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

	def validate_coordinates(self, array_shape, cords):
		"""

		:param array_shape:
		:param cords:
		:return:
		"""
		try:
			if len(array_shape) > 2:
				array_shape = [array_shape[0], array_shape[1]]
			if (cords[0] < 0) or (cords[0] >= array_shape[0]) or (cords[1] < 0) or (cords[1] >= array_shape[1]):
				return False
			else:
				return True
		except Exception as CordValidationException:
			print("Exception while running function 'validate_coordinates'.\n", CordValidationException)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
