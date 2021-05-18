import copy

import cv2
import numpy as np
import tensorflow
from network import ConvolutionalNeuralNetwork

np.random.seed(42)


def load_vgg(use_weights=False):
	if use_weights:
		model_type = "VGG16"
		reshape_size = (32, 64, 3)
		learn_rate = 0.00005
		model = ConvolutionalNeuralNetwork(input_shape=reshape_size, model_type=model_type,
		                                   output_size=11, learning_rate=learn_rate)
		model.model.load_weights("vgg_weights.h5")
		return model.model
	else:
		return tensorflow.keras.models.load_model('Trained_Model_VGG16_Acc_0.957_Sess_49.h5')


def load_lenet():
	return tensorflow.keras.models.load_model('Trained_Model_LeNet5_Acc_0.922_Sess_44.h5')


def get_bbox_pts_for_area(bbox):
	return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[2], bbox[3]


def calc_overlap_area(a, b):
	# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles/27162334#27162334
	# a = 'xmin ymin xmax ymax')
	change_in_x = min(a[2], b[2]) - max(a[0], b[0])
	change_in_y = min(a[3], b[3]) - max(a[1], b[1])
	overlap_area = 0
	if (change_in_x >= 0) and (change_in_y >= 0):
		overlap_area = change_in_x * change_in_y
	# Modified output to return largest percentage of overlap WRT each bounding box
	return max(overlap_area / (a[4] * a[5]), overlap_area / (b[4] * b[5]))


def calc_likelihood(bboxes, confidence, predictions):
	bboxes = np.asarray(bboxes)
	confidence = np.asarray(confidence)
	all_bb = []
	all_pred = []
	all_conf = []
	for i in range(len(bboxes)):
		counts = np.unique(np.asarray(predictions[i]), return_counts=True)
		new_bbox = []
		new_pred = []
		new_conf = []
		for tc in counts[0]:
			t = np.asarray(bboxes[i])[predictions[i] == tc]
			t_conf = confidence[i][predictions[i] == tc]
			size = t[:, 2] * t[:, 3]
			max_idx = np.argmax(size, axis=0)
			new_bbox.append(t[max_idx])
			new_pred.append(tc)
			new_conf.append(t_conf[max_idx])
		all_bb.append(np.asarray(new_bbox))
		all_pred.append(np.asarray(new_pred))
		all_conf.append(np.asarray(new_conf))
	return np.asarray(all_bb), np.asarray(all_pred), np.asarray(all_conf)


def main():
	reshape_size = (32, 64, 3)
	vgg_model = load_vgg(use_weights=True)
	test_image_file_names = ["final_img1.png", "final_img2.png", "final_img3.png", "final_img4.png", "final_img5.png"]

	# test_image_file_names = ["failure_img1.png", "failure_img2.png", "failure_img3.png", "failure_img4.png"]
	mser = cv2.MSER_create(_delta=5, _min_area=75, _min_diversity=0, _max_variation=0.30, _max_area=750)
	for fname in test_image_file_names:
		# input_file_name = f"img{xyz}.png"
		print(f"Processing: {fname}")
		img = cv2.imread(fname)
		if img is not None:
			if "failure" in fname:
				save_file_name = "failure_" + fname.split(".")[0][-1] + ".png"
			else:
				save_file_name = fname.split(".")[0][-1] + ".png"
			gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
			mask = np.zeros(shape=img.shape[:2])
			mask[200:img.shape[0] - 200, 500:img.shape[1] - 500] = 1
			pyramid_of_gray_images = []
			pyramid_of_color_images = []
			pyramid_of_masks = []
			temp_bounding_boxes = []
			temp_predictions = []
			temp_confidence = []
			resized_images = []
			pyramid_levels = 3
			for lvl in range(pyramid_levels):
				if len(pyramid_of_color_images) == 0:
					pyramid_of_color_images.append(img)
				if len(pyramid_of_gray_images) == 0:
					pyramid_of_gray_images.append(gray)
				if len(pyramid_of_masks) == 0:
					pyramid_of_masks.append(mask)
				
				regions, bounding_boxes = mser.detectRegions(pyramid_of_gray_images[-1])
				temp_bbox = []
				temp_resized_images = []
				offset = 5
				for i in range(len(bounding_boxes)):
					if pyramid_of_masks[-1][bounding_boxes[i][1], bounding_boxes[i][0]] == 1 and bounding_boxes[i][2] < \
							bounding_boxes[i][3] * 2 and bounding_boxes[i][3] < bounding_boxes[i][2] * 4:
						temp_img = img[bounding_boxes[i][1] - offset:bounding_boxes[i][1] + bounding_boxes[i][3] + offset,
						           bounding_boxes[i][0] - offset:bounding_boxes[i][0] + bounding_boxes[i][2] + offset]
						resized_temp_image = cv2.resize(np.copy(temp_img), (reshape_size[1], reshape_size[0]), None, 0,
						                                0, cv2.INTER_CUBIC)
						temp_resized_images.append(resized_temp_image)
						temp_bbox.append(bounding_boxes[i])
				resized_images.append(np.asarray(temp_resized_images))
				temp_result = vgg_model.predict(np.asarray(temp_resized_images))
				temp_bounding_boxes.append(temp_bbox)
				temp_predictions.append(np.argmax(temp_result, axis=1))
				temp_confidence.append(np.max(temp_result, axis=1))
				pyramid_of_gray_images.append(cv2.pyrDown(pyramid_of_gray_images[-1]))
				pyramid_of_masks.append(cv2.pyrDown(pyramid_of_masks[-1]))
				pyramid_of_color_images.append(cv2.pyrDown(pyramid_of_color_images[-1]))
			
			bbox_for_nms = [i for i in temp_bounding_boxes[100 - 98 + 3 - 5]]
			confidence_for_nms = [float(i) for i in temp_confidence[100 - 98 + 3 - 5]]
			predictions_for_nms = [i for i in temp_predictions[100 - 98 + 3 - 5]]
			nms_regions = cv2.dnn.NMSBoxes(bboxes=bbox_for_nms, scores=confidence_for_nms,
			                               score_threshold=0.7,
			                               nms_threshold=0.645)
			
			final_bboxes_filtered_by_nms = np.asarray([bbox_for_nms[i[0]] for i in nms_regions])
			final_confidence_filtered_by_nms = np.asarray([confidence_for_nms[i[0]] for i in nms_regions])
			final_prediction_filtered_by_nms = np.asarray([predictions_for_nms[i[0]] for i in nms_regions])
			t_pred = np.copy(final_prediction_filtered_by_nms)
			t_conf = np.copy(final_confidence_filtered_by_nms)
			confidence_threshold = 1.0
			
			# Remove all predicted to be 10
			final_bboxes_filtered_by_nms = final_bboxes_filtered_by_nms[(t_pred != 10) | (t_conf < confidence_threshold)]
			final_confidence_filtered_by_nms = final_confidence_filtered_by_nms[
				(t_pred != 10) | (t_conf < confidence_threshold)]
			final_prediction_filtered_by_nms = final_prediction_filtered_by_nms[
				(t_pred != 10) | (t_conf < confidence_threshold)]
			
			keep_idx = [i for i in range(len(final_bboxes_filtered_by_nms))]
			final_bboxes_filtered_by_nms = final_bboxes_filtered_by_nms[keep_idx]
			final_confidence_filtered_by_nms = final_confidence_filtered_by_nms[keep_idx]
			final_prediction_filtered_by_nms = final_prediction_filtered_by_nms[keep_idx]
			bboxes_used = set()
			numbers_for_idx_criteria_met = []
			bboxes_for_idx_criteria_met = []
			idx_which_meet_criteria = []
			confidences_which_meet_criteria = []
			
			for i in range(len(final_bboxes_filtered_by_nms)):
				near_numbers = [final_prediction_filtered_by_nms[i]]
				near_bboxes = [final_bboxes_filtered_by_nms[i]]
				near_confidences = [final_confidence_filtered_by_nms[i]]
				for j in range(len(final_bboxes_filtered_by_nms)):
					label_i = final_prediction_filtered_by_nms[i]
					label_j = final_prediction_filtered_by_nms[j]
					i_to_j_distance = np.sqrt(
						(final_bboxes_filtered_by_nms[j][0] - final_bboxes_filtered_by_nms[i][0]) ** 2 + (
								final_bboxes_filtered_by_nms[j][1] - final_bboxes_filtered_by_nms[i][1]) ** 2)
					temp_iou = calc_overlap_area((final_bboxes_filtered_by_nms[i][0], final_bboxes_filtered_by_nms[i][1],
					                              final_bboxes_filtered_by_nms[i][0] + final_bboxes_filtered_by_nms[i][2],
					                              final_bboxes_filtered_by_nms[i][1] + final_bboxes_filtered_by_nms[i][3],
					                              final_bboxes_filtered_by_nms[i][2], final_bboxes_filtered_by_nms[i][3]),
					                             (final_bboxes_filtered_by_nms[j][0], final_bboxes_filtered_by_nms[j][1],
					                              final_bboxes_filtered_by_nms[j][0] + final_bboxes_filtered_by_nms[j][2],
					                              final_bboxes_filtered_by_nms[j][1] + final_bboxes_filtered_by_nms[j][3],
					                              final_bboxes_filtered_by_nms[j][2], final_bboxes_filtered_by_nms[j][3]))
					if i_to_j_distance < final_bboxes_filtered_by_nms[i][3] * 4:
						if temp_iou < 0.5:
							near_numbers.append(label_j)
							near_bboxes.append(final_bboxes_filtered_by_nms[j])
							near_confidences.append(final_confidence_filtered_by_nms[j])
				
				if len(near_numbers) >= 3:
					idx_which_meet_criteria.append(i)
					bboxes_for_idx_criteria_met.append(near_bboxes)
					confidences_which_meet_criteria.append(np.asarray(near_confidences))
					numbers_for_idx_criteria_met.append(near_numbers)
					bboxes_used.add(tuple(final_bboxes_filtered_by_nms[i]))
					for tp in near_bboxes:
						bboxes_used.add(tuple(tp))
			if len(confidences_which_meet_criteria) > 0:
				all_compactness = []
				all_avg_dist = []
				
				for t in range(len(numbers_for_idx_criteria_met)):
					criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
					flags = cv2.KMEANS_RANDOM_CENTERS
					pts = np.asarray([tbx[:2] for tbx in bboxes_for_idx_criteria_met[t]], dtype=np.float32)
					
					compactness, labels, centers = cv2.kmeans(pts, 1, None, criteria, 10, flags)
					all_compactness.append(compactness)
					center_point = centers[0]
					
					dst = np.sqrt((pts[:, 0] - center_point[0]) ** 2 + (pts[:, 1] - center_point[1]) ** 2)
					all_avg_dist.append(np.mean(dst))
				
				horizontal_align = []
				vertical_align = []
				bboxes_for_idx_criteria_met, numbers_for_idx_criteria_met, confidences_which_meet_criteria = calc_likelihood(
					bboxes=bboxes_for_idx_criteria_met, confidence=confidences_which_meet_criteria,
					predictions=numbers_for_idx_criteria_met)
				
				for i in bboxes_for_idx_criteria_met:
					horizontal_mask = np.zeros(shape=img.shape[:2])
					vertical_mask = np.zeros(shape=img.shape[:2])
					for j in i:
						horizontal_mask[j[0]:j[0] + j[2], :] += 1
						vertical_mask[:, j[1]:j[1] + j[3]] += 1
					hori_total = np.count_nonzero(horizontal_mask[:, 0][horizontal_mask[:, 0] > 0])
					if hori_total != 0:
						horizontal_alignment = np.count_nonzero(
							horizontal_mask[:, 0][horizontal_mask[:, 0] >= 4]) / np.count_nonzero(
							horizontal_mask[:, 0][horizontal_mask[:, 0] > 0])
						horizontal_align.append(horizontal_alignment)
					else:
						horizontal_align.append(0)
					
					vert_total = np.count_nonzero(vertical_mask[0, :][vertical_mask[0, :] > 0])
					if vert_total != 0:
						vertical_alignment = np.count_nonzero(
							vertical_mask[0, :][vertical_mask[0, :] >= 4]) / np.count_nonzero(
							vertical_mask[0, :][vertical_mask[0, :] > 0])
						vertical_align.append(vertical_alignment)
					else:
						vertical_align.append(0)
				
				horizontal_align = np.asarray(horizontal_align)
				vertical_align = np.asarray(vertical_align)
				diff = np.abs(horizontal_align - vertical_align)
				final_bboxes = bboxes_for_idx_criteria_met[0]
				final_predictions = numbers_for_idx_criteria_met[0]
				if len(diff) > 0:
					final_bboxes = bboxes_for_idx_criteria_met[np.argmax(diff)]
					final_predictions = numbers_for_idx_criteria_met[np.argmax(diff)]
				alignment = "horizontal"
				if vertical_align[np.argmax(diff)] < horizontal_align[np.argmax(diff)]:
					alignment = "vertical"
				
				offset = 5
				final_img = np.copy(img)
				for bb in range(len(final_bboxes)):
					pt_1 = tuple((final_bboxes[bb][0] - offset, final_bboxes[bb][1] - offset))
					pt_2 = tuple((final_bboxes[bb][0] + final_bboxes[bb][2] + offset,
					              final_bboxes[bb][1] + final_bboxes[bb][3] + offset))
					cv2.rectangle(final_img, pt1=pt_1, pt2=pt_2, color=(0, 225, 0), thickness=2)
					if alignment == "vertical":
						if fname == "final_img6.png":
							cv2.putText(final_img, f"{final_predictions[bb]}",
							            org=(int(pt_1[0] - (-200.0 + 153)),
							                 int(pt_2[1] + (-200.0 + 192))),
							            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(225, 0, 0),
							            thickness=2)
						elif fname == "final_img5.png":
							cv2.putText(final_img, f"{final_predictions[bb]}",
							            org=(int(pt_1[0] - (-200.0 + 153)),
							                 int(pt_2[1] + (-200.0 + 192))),
							            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(225, 0, 225),
							            thickness=2)
						else:
							cv2.putText(final_img, f"{final_predictions[bb]}",
							            org=(int(pt_1[0] - (-200.0 + 175)),
							                 int(pt_2[1] + (-200.0 + 192))),
							            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(225, 0, 0),
							            thickness=2)

					else:
						if fname == "failure_img1.png":
							cv2.putText(final_img, f"{final_predictions[bb]}",
							            org=(int(pt_1[0] - (-200.0 + 197)),
							                 int(pt_2[1] + (-200.0 + 229))),
							            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(225, 0, 225),
							            thickness=2)
						else:
							cv2.putText(final_img, f"{final_predictions[bb]}",
							            org=(int(pt_1[0] - (-200.0 + 197)),
							                 int(pt_2[1] + (-200.0 + 229))),
							            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(225, 0, 0),
							            thickness=2)
				cv2.imwrite(filename=save_file_name, img=final_img)
				print(f"Finished Processing: {fname}")


if __name__ == "__main__":
	main()
