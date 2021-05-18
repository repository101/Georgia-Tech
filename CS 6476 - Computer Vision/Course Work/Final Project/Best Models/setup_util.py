import os
import sys
import time
import numpy as np
import cv2
import glob
import pickle
import os.path


def get_sub_image_name(img_name, number):
	name_parts = img_name.split(".")
	prefix = name_parts[0]
	suffix = "." + name_parts[-1]
	return prefix + "_" + str(number) + suffix


def setup_process_images(num=1000):
	is_training = True
	is_testing = False
	is_extra = False
	
	all_data = {"Training": {}, "Testing": {}, "Extra": {}}
	
	with open("train_digitStruct.mat.pkl", "rb") as input_file:
		data = pickle.load(input_file)
	prev_pct = 0
	files = glob.glob(f"{os.getcwd()}/training images/*.png")
	names_as_array = np.asarray(data["names"])
	num_files = names_as_array.shape[0]
	# Problems
	#   Cannot create array of images due to images being different sizes
	processed_images = set()
	print("Beginning to load Training set")
	count = 0
	for file in files[:num]:
		image_name = file.split("\\")[-1]
		processed_images.add(image_name)
		if is_training:
			key = "Training"
		elif is_testing:
			key = "Testing"
		elif is_extra:
			key = "Extra"
		else:
			key = None
		
		file_path = file[:len(file) - len(image_name)]
		if "_" not in image_name:
			struct_idx = np.argwhere(names_as_array == image_name)[0][0]
			bounding_box_from_struct = data["bboxes"][struct_idx]
			if np.abs(bounding_box_from_struct[0] - bounding_box_from_struct[1]) > 0 and np.abs(
					bounding_box_from_struct[2] - bounding_box_from_struct[3]) > 1:
				all_data[key][image_name] = {"Original Combined Image": None,
				                             "Grayscale Combined Image": None,
				                             "Numbers in Combined Image": None,
				                             "Bounding Box in Combined Image": None,
				                             "Image with Bounding Box": None,
				                             "Sub Images": None}
				
				name_from_struct = data["names"][struct_idx]
				label_from_struct = data["labels"][struct_idx]
				number_of_numbers_in_image = label_from_struct[0]
				the_numbers_in_image = label_from_struct[1:number_of_numbers_in_image + 1]
				bounding_box_from_struct = data["bboxes"][struct_idx]
				orig_image = cv2.imread(file)
				all_data[key][image_name]["Original Combined Image"] = np.copy(orig_image)
				all_data[key][image_name]["Grayscale Combined Image"] = cv2.cvtColor(np.copy(orig_image),
				                                                                     cv2.COLOR_BGR2GRAY)
				all_data[key][image_name]["Numbers in Combined Image"] = the_numbers_in_image
				all_data[key][image_name]["Bounding Box in Combined Image"] = bounding_box_from_struct
				a, b, c, d = bounding_box_from_struct
				
				all_data[key][image_name]["Image with Bounding Box"] = cv2.rectangle(np.copy(orig_image),
				                                                                     pt1=(bounding_box_from_struct[0],
				                                                                          bounding_box_from_struct[1]),
				                                                                     pt2=(bounding_box_from_struct[2],
				                                                                          bounding_box_from_struct[3]),
				                                                                     color=(0, 0, 255), thickness=2)
				if len(the_numbers_in_image) > 0:
					all_data[key][image_name]["Sub Images"] = {}
					all_data[key][image_name]["Grayscale Sub Images"] = {}
					for temp_num in the_numbers_in_image:
						all_data[key][image_name]["Sub Images"][temp_num] = cv2.imread(
							f"{file_path}{get_sub_image_name(image_name, temp_num)}")
						all_data[key][image_name]["Grayscale Sub Images"][temp_num] = cv2.cvtColor(
							np.copy(all_data[key][image_name]["Sub Images"][temp_num]), cv2.COLOR_BGR2GRAY)
				count += 1
		pct = int(count // (num_files / 100))
		if pct > prev_pct:
			print(f"Currently Processed: {count}")
			print(f"Remaining: {num_files - count}")
			t_string = "[" + ("#" * pct) + ((100 - pct) * " ") + " ]"
			print(t_string)
			prev_pct = pct
	
	print("Finished loading Training set\n")
	print("Saving updated object")
	with open("ALL_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data, output_file, protocol=-1)
	with open("ALL_TRAINING_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data["Training"], output_file, protocol=-1)
	print("Finished saving updated object\n")
	prev_pct = 0
	is_training = False
	is_testing = True
	is_extra = False
	
	with open("test_digitStruct.mat.pkl", "rb") as input_file:
		data = pickle.load(input_file)
	prev_pct = 0
	count = 0
	temp_num_files = 0
	files = glob.glob(f"{os.getcwd()}/testing images/*.png")
	names_as_array = np.asarray(data["names"])
	num_files = names_as_array.shape[0]
	# Problems
	#   Cannot create array of images due to images being different sizes
	print("Beginning to load Testing set")
	processed_images = set()
	count = 0
	for file in files[:num]:
		image_name = file.split("\\")[-1]
		processed_images.add(image_name)
		if is_training:
			key = "Training"
		elif is_testing:
			key = "Testing"
		elif is_extra:
			key = "Extra"
		else:
			key = None
		
		file_path = file[:len(file) - len(image_name)]
		if "_" not in image_name:
			struct_idx = np.argwhere(names_as_array == image_name)[0][0]
			bounding_box_from_struct = data["bboxes"][struct_idx]
			if np.abs(bounding_box_from_struct[0] - bounding_box_from_struct[1]) > 0 and np.abs(
					bounding_box_from_struct[2] - bounding_box_from_struct[3]) > 1:
				all_data[key][image_name] = {"Original Combined Image": None,
				                             "Grayscale Combined Image": None,
				                             "Numbers in Combined Image": None,
				                             "Bounding Box in Combined Image": None,
				                             "Image with Bounding Box": None,
				                             "Sub Images": None}
				
				name_from_struct = data["names"][struct_idx]
				label_from_struct = data["labels"][struct_idx]
				number_of_numbers_in_image = label_from_struct[0]
				the_numbers_in_image = label_from_struct[1:number_of_numbers_in_image + 1]
				bounding_box_from_struct = data["bboxes"][struct_idx]
				orig_image = cv2.imread(file)
				all_data[key][image_name]["Original Combined Image"] = np.copy(orig_image)
				all_data[key][image_name]["Grayscale Combined Image"] = cv2.cvtColor(np.copy(orig_image),
				                                                                     cv2.COLOR_BGR2GRAY)
				all_data[key][image_name]["Numbers in Combined Image"] = the_numbers_in_image
				all_data[key][image_name]["Bounding Box in Combined Image"] = bounding_box_from_struct
				a, b, c, d = bounding_box_from_struct
				
				all_data[key][image_name]["Image with Bounding Box"] = cv2.rectangle(np.copy(orig_image),
				                                                                     pt1=(bounding_box_from_struct[0],
				                                                                          bounding_box_from_struct[1]),
				                                                                     pt2=(bounding_box_from_struct[2],
				                                                                          bounding_box_from_struct[3]),
				                                                                     color=(0, 0, 255), thickness=2)
				if len(the_numbers_in_image) > 0:
					all_data[key][image_name]["Sub Images"] = {}
					all_data[key][image_name]["Grayscale Sub Images"] = {}
					for temp_num in the_numbers_in_image:
						all_data[key][image_name]["Sub Images"][temp_num] = cv2.imread(
							f"{file_path}{get_sub_image_name(image_name, temp_num)}")
						all_data[key][image_name]["Grayscale Sub Images"][temp_num] = cv2.cvtColor(
							np.copy(all_data[key][image_name]["Sub Images"][temp_num]), cv2.COLOR_BGR2GRAY)
				count += 1
		pct = int(count // (num_files / 100))
		if pct > prev_pct:
			print(f"Currently Processed: {count}")
			print(f"Remaining: {num_files - count}")
			t_string = "[" + ("#" * pct) + ((100 - pct) * " ") + " ]"
			print(t_string)
			prev_pct = pct
	
	print("Finished loading Testing set\n")
	print("Saving updated object")
	with open("ALL_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data, output_file, protocol=-1)
	with open("ALL_TESTING_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data["Testing"], output_file, protocol=-1)
	print("Finished saving updated object\n")
	
	prev_pct = 0
	count = 0
	num_files = 0
	is_training = False
	is_testing = False
	is_extra = True
	
	with open("extra_digitStruct.mat.pkl", "rb") as input_file:
		data = pickle.load(input_file)
	prev_pct = 0
	files = glob.glob(f"{os.getcwd()}/extra images/*.png")
	names_as_array = np.asarray(data["names"])
	num_files = names_as_array.shape[0]
	# Problems
	#   Cannot create array of images due to images being different sizes
	print("Beginning to load Extra set")
	processed_images = set()
	count = 0
	for file in files[:num]:
		image_name = file.split("\\")[-1]
		processed_images.add(image_name)
		if is_training:
			key = "Training"
		elif is_testing:
			key = "Testing"
		elif is_extra:
			key = "Extra"
		else:
			key = None
		
		file_path = file[:len(file) - len(image_name)]
		if "_" not in image_name:
			struct_idx = np.argwhere(names_as_array == image_name)[0][0]
			bounding_box_from_struct = data["bboxes"][struct_idx]
			if np.abs(bounding_box_from_struct[0] - bounding_box_from_struct[1]) > 0 and np.abs(
					bounding_box_from_struct[2] - bounding_box_from_struct[3]) > 1:
				all_data[key][image_name] = {"Original Combined Image": None,
				                             "Grayscale Combined Image": None,
				                             "Numbers in Combined Image": None,
				                             "Bounding Box in Combined Image": None,
				                             "Image with Bounding Box": None,
				                             "Sub Images": None}
				
				name_from_struct = data["names"][struct_idx]
				label_from_struct = data["labels"][struct_idx]
				number_of_numbers_in_image = label_from_struct[0]
				the_numbers_in_image = label_from_struct[1:number_of_numbers_in_image + 1]
				bounding_box_from_struct = data["bboxes"][struct_idx]
				orig_image = cv2.imread(file)
				all_data[key][image_name]["Original Combined Image"] = np.copy(orig_image)
				all_data[key][image_name]["Grayscale Combined Image"] = cv2.cvtColor(np.copy(orig_image),
				                                                                     cv2.COLOR_BGR2GRAY)
				if image_name == "127575.png":
					the_numbers_in_image = (4, 0, 3)
				elif image_name == "199576.png":
					the_numbers_in_image = (2, 9, 0)
				
				all_data[key][image_name]["Numbers in Combined Image"] = the_numbers_in_image
				all_data[key][image_name]["Bounding Box in Combined Image"] = bounding_box_from_struct
				a, b, c, d = bounding_box_from_struct
				
				all_data[key][image_name]["Image with Bounding Box"] = cv2.rectangle(np.copy(orig_image),
				                                                                     pt1=(bounding_box_from_struct[0],
				                                                                          bounding_box_from_struct[1]),
				                                                                     pt2=(bounding_box_from_struct[2],
				                                                                          bounding_box_from_struct[3]),
				                                                                     color=(0, 0, 255), thickness=2)
				if len(the_numbers_in_image) > 0:
					all_data[key][image_name]["Sub Images"] = {}
					all_data[key][image_name]["Grayscale Sub Images"] = {}
					for temp_num in the_numbers_in_image:
						try:
							all_data[key][image_name]["Sub Images"][temp_num] = cv2.imread(
								f"{file_path}{get_sub_image_name(image_name, temp_num)}")
							all_data[key][image_name]["Grayscale Sub Images"][temp_num] = cv2.cvtColor(
								np.copy(all_data[key][image_name]["Sub Images"][temp_num]), cv2.COLOR_BGR2GRAY)
						except Exception as err:
							img1 = cv2.imread(f"{file_path}{get_sub_image_name(image_name, temp_num)}")
							img2 = cv2.imread(f"{file_path}{get_sub_image_name(image_name, temp_num)}")
							print(f"File Name: {image_name}")
							print(f"Missing Sub Image: {temp_num}")
				
				count += 1
		pct = int(count // (num_files / 100))
		if pct > prev_pct:
			print(f"Currently Processed: {count}")
			print(f"Remaining: {num_files - count}")
			t_string = "[" + ("#" * pct) + ((100 - pct) * " ") + " ]"
			print(t_string)
			prev_pct = pct
	print("Finished loading Extra set\n")
	print("Saving updated object")
	with open("ALL_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data, output_file, protocol=-1)
	with open("ALL_EXTRA_DATA.pkl", "wb") as output_file:
		pickle.dump(all_data["Extra"], output_file, protocol=-1)
	print("Finished saving updated object\n")
	
	return all_data


if __name__ == "__main__":
	data = setup_process_images(num=-1)

