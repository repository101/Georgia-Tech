import pickle
import cv2
import numpy as np
import glob


def load_data(name):
	if name.lower() == "all":
		with open("ALL_DATA.pkl", "rb") as input_file:
			data = pickle.load(input_file)
			
	elif name.lower() == "train" or name.lower() == "training":
		with open("ALL_TRAINING_DATA.pkl", "rb") as input_file:
			data = pickle.load(input_file)
			
	elif name.lower() == "test" or name.lower() == "testing":
		with open("ALL_TESTING_DATA.pkl", "rb") as input_file:
			data = pickle.load(input_file)
			
	elif name.lower() == "extra":
		with open("ALL_EXTRA_DATA.pkl", "rb") as input_file:
			data = pickle.load(input_file)
	return data


def df_to_array(df, img_shape):
	labels = df.iloc[:, -1].to_numpy()
	images = np.reshape(df.iloc[:, :-1].to_numpy(), newshape=(labels.shape[0], img_shape[0], img_shape[1], 3))
	return images, labels


def images_from_data_dict(data_dict, which_images="sub", get_grayscale=False, reshape_size=(16, 32),
                          number_to_load=[0,5000], load_negatives=False, loc="training images", blur_images=False,
                          blur_filter_size=3):
	if which_images == "sub":
		if get_grayscale:
			img__key = "Grayscale Sub Images"
		else:
			img_key = "Sub Images"
		keys_for_dict_based_on_number_to_load = np.asarray(list(data_dict.keys()))
		keys_for_dict_based_on_number_to_load = keys_for_dict_based_on_number_to_load[number_to_load[0]: number_to_load[1]]
		images = None
		prev_pct = 0
		count = 0
		files = set([i.split("\\")[-1] for i in glob.glob(f"all_data/{loc}/*.png")])
		num_files = keys_for_dict_based_on_number_to_load.shape[0]
		for key, val in data_dict.items():
			if key in keys_for_dict_based_on_number_to_load:
				for key1, val1 in val[img_key].items():
					bbox = val["Bounding Box in Combined Image"]
					filename_parts = key.split(".")
					sub_temp_image_filename = filename_parts[0] + "_" + str(key1) + "." + filename_parts[-1]
					if sub_temp_image_filename not in files:
						continue
					else:
						val1[bbox[0]:bbox[1], bbox[2]:bbox[3]] = (-1,-1,-1)
						if (val1.shape[0] * val1.shape[1]) > 150:
							if val1 is not None:
								label = key1
								if key1 == 10:
									label = 0
								if val1.shape[:2] != reshape_size:
								# if val1.shape[:2] != reshape_size:
									temp_img = cv2.resize(np.copy(val1), (reshape_size[0], reshape_size[1]), None, 0, 0, cv2.INTER_CUBIC)
									if blur_images:
										temp_img = cv2.GaussianBlur(temp_img, ksize=(blur_filter_size,blur_filter_size),
										                            sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT_101)
									if images is None:
										images = []
										temp_flattened = list(np.ravel(temp_img))
										temp_flattened.append(label)
										images.append(temp_flattened)
									else:
										temp_flattened = list(np.ravel(temp_img))
										temp_flattened.append(label)
										images.append(temp_flattened)
								else:
									if images is None:
										images = []
									else:
										temp_img = np.copy(val1)
										if blur_images:
											temp_img = cv2.GaussianBlur(temp_img, ksize=(blur_filter_size,blur_filter_size),
											                            sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT_101)
											
										temp_flattened = list(np.ravel(temp_img))
										temp_flattened.append(label)
									images.append(temp_flattened)
							if load_negatives:
								bbox = val["Bounding Box in Combined Image"]
								temp_img = np.copy(val["Original Combined Image"])
								cv2.rectangle(temp_img, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(1, 2, 3), thickness=-1)
								num_of_splits = 4
								count = 2
								for i in np.arange(0, temp_img.shape[0], temp_img.shape[0] // num_of_splits):
									for j in np.arange(0, temp_img.shape[1], temp_img.shape[1] // num_of_splits):
										if i > temp_img.shape[0]:
											continue
										elif j > temp_img.shape[1]:
											continue
										elif (i + temp_img.shape[0] // 8) > temp_img.shape[0] - 1:
											continue
										elif (j + temp_img.shape[1] // 8) > temp_img.shape[1] - 1:
											continue
										else:
											sub_temp_image = temp_img[i:i + temp_img.shape[0] // num_of_splits,
											                 j:j + temp_img.shape[1] // num_of_splits]
											check = np.any(sub_temp_image[:, :, [0, 1, 2]] == (1, 2, 3))
											if check:
												continue
											else:
												if count <= 2:
													sub_temp_image = cv2.resize(np.copy(sub_temp_image), (reshape_size[0], reshape_size[1]),
													                            None, 0, 0, cv2.INTER_CUBIC)
													sub_temp_image = list(np.ravel(sub_temp_image))
													sub_temp_image.append(10)
													images.append(sub_temp_image)
													count += 1
												else:
													continue
				count += 1
				pct = int(count // (num_files / 100))
				if pct > prev_pct:
					print(f"Remaining: {num_files - count}")
					t_string = "[" + ("#" * pct) + ((100 - pct) * " ") + " ]"
					print(t_string)
					prev_pct = pct
	
		return np.asarray(images)
	
	
def evenly_distribute_classes(data_x, data_y):
	unique_vals, unique_counts = np.unique(data_y, return_counts=True)
	min_count = np.min(unique_counts)
	idx_tracker = []
	for i in range(0, 11):
		idx = np.argwhere(data_y == i)[:, 0]
		temp_idx = np.random.choice(idx, min_count, replace=False)
		idx_tracker.append(temp_idx)
		
	idx_tracker = np.ravel(np.asarray(idx_tracker))
	np.random.shuffle(idx_tracker)
	return data_x[idx_tracker], data_y[idx_tracker]
