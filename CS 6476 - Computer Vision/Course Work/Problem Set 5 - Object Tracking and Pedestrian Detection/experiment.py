"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5
from ps5_utils import visualize_filter, visualize_filter_pt_5, run_kalman_filter_pt_5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
	print("Part 1b")

	template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
	save_frames = {
		12: os.path.join(output_dir, 'ps5-1-b-1.png'),
		28: os.path.join(output_dir, 'ps5-1-b-2.png'),
		57: os.path.join(output_dir, 'ps5-1-b-3.png'),
		97: os.path.join(output_dir, 'ps5-1-b-4.png')
	}
	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
				os.path.join(input_dir, "circle"))


def part_1c():
	print("Part 1c")

	template_loc = {'x': 311, 'y': 217}
	save_frames = {
		12: os.path.join(output_dir, 'ps5-1-c-1.png'),
		30: os.path.join(output_dir, 'ps5-1-c-2.png'),
		81: os.path.join(output_dir, 'ps5-1-c-3.png'),
		155: os.path.join(output_dir, 'ps5-1-c-4.png')
	}

	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
				os.path.join(input_dir, "walking"))


def part_2a():

	template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

	save_frames = {
		8: os.path.join(output_dir, 'ps5-2-a-1.png'),
		28: os.path.join(output_dir, 'ps5-2-a-2.png'),
		57: os.path.join(output_dir, 'ps5-2-a-3.png'),
		97: os.path.join(output_dir, 'ps5-2-a-4.png')
	}
	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_2a(
		ps5.ParticleFilter,  # particle filter model class
		template_loc,
		save_frames,
		os.path.join(input_dir, "circle"))


def part_2b():

	template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

	save_frames = {
		12: os.path.join(output_dir, 'ps5-2-b-1.png'),
		28: os.path.join(output_dir, 'ps5-2-b-2.png'),
		57: os.path.join(output_dir, 'ps5-2-b-3.png'),
		97: os.path.join(output_dir, 'ps5-2-b-4.png')
	}
	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_2b(
		ps5.ParticleFilter,  # particle filter model class
		template_loc,
		save_frames,
		os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
	template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

	save_frames = {
		20: os.path.join(output_dir, 'ps5-3-a-1.png'),
		48: os.path.join(output_dir, 'ps5-3-a-2.png'),
		158: os.path.join(output_dir, 'ps5-3-a-3.png')
	}
	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_3(
		ps5.AppearanceModelPF,  # particle filter model class
		template_rect,
		save_frames,
		os.path.join(input_dir, "pres_debate"))


def part_4():
	template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

	save_frames = {
		40: os.path.join(output_dir, 'ps5-4-a-1.png'),
		100: os.path.join(output_dir, 'ps5-4-a-2.png'),
		240: os.path.join(output_dir, 'ps5-4-a-3.png'),
		300: os.path.join(output_dir, 'ps5-4-a-4.png')
	}
	# Define process and measurement arrays if you want to use other than the
	# default.
	ps5.part_4(
		ps5.MDParticleFilter,  # particle filter model class
		template_rect,
		save_frames,
		os.path.join(input_dir, "pedestrians"))


def part_5():
	"""Tracking multiple Targets.

	Use either a Kalman or particle filter to track multiple targets
	as they move through the given video.  Use the sequence of images
	in the TUD-Campus directory.

	Follow the instructions in the problem set instructions.

	Place all your work in this file and this section.
	"""
	save_video = False
	if save_video:
		width = 1330
		height = 480
		FPS = 10
		seconds = 10
		fourcc = cv2.VideoWriter_fourcc(*'MP42')
		video = cv2.VideoWriter('./pedestrian_tracker.mp4', fourcc, float(FPS), (width, height))
	save_frames = {
		29: os.path.join(output_dir, 'ps5-5-a-1.png'),
		56: os.path.join(output_dir, 'ps5-5-b-2.png'),
		71: os.path.join(output_dir, 'ps5-5-c-3.png'),
	}
	# Just pedestrian 1 face
	pedestrian_1_template_rect = {'x': 115, 'y': 160, 'w': 35, 'h': 40}
	
	# pedestrian_1_template_rect = {'x': 95, 'y': 142, 'w': 50, 'h': 200}
	pedestrian_2_template_rect = {'x': 305, 'y': 195, 'w': 25, 'h': 30}
	wait_val = 1
	# pedestrian_2_template_rect = {'x': 280, 'y': 195, 'w': 50, 'h': 125}
	pedestrian_3_enter_frame = 28
	# pedestrian_3_template_rect = {'x': 35, 'y': 175, 'w': 20, 'h': 75}
	
	# Just pedestrian 3 face {'x': 20, 'y': 175, 'w': 40, 'h': 40}
	pedestrian_3_template_rect = {'x': 20, 'y': 175, 'w': 40, 'h': 40}
	
	# Define process and measurement arrays if you want to use other than the
	# default
	
	imgs_dir = os.path.join(input_dir, "TUD-Campus")
	imgs_list = [f for f in os.listdir(imgs_dir)
				 if f[0] != '.' and f.endswith('.jpg')]
	imgs_list.sort()
	pedestrian_1_filter = None
	pedestrian_1_template = None
	pedestrian_1_rect_color = (40, 39, 214)
	pedestrian_2_filter = None
	pedestrian_2_template = None
	pedestrian_2_rect_color = (180, 119, 31)
	pedestrian_3_filter = None
	pedestrian_3_template = None
	pedestrian_3_rect_color = (44, 160, 44)
	
	which_peds = {1,2,3}

	# Initialize objects
	
	frame_num = 1
	for img in imgs_list:
		frame = cv2.imread(os.path.join(imgs_dir, img))
		ped_1_frame = np.copy(frame)
		ped_2_frame = np.copy(frame)
		ped_3_frame = np.copy(frame)
		if frame_num >= 29:
			wait_val = 1
		
		# region Kalman Filtering ALL FAILED
		# Q = 0.1 * np.eye(4)  # Process noise array
		# R = 0.1 * np.eye(2)  # Measurement noise array
		# NOISE_2 = {'x': 7.5, 'y': 7.5}
		# if pedestrian_1_filter is None:
		#     pedestrian_1_filter = ps5.KalmanFilter(pedestrian_1_template_rect['x'], pedestrian_1_template_rect['y'], Q, R)
		#
		# sensor = "matching"
		#
		# if sensor == "hog":
		#     hog = cv2.HOGDescriptor()
		#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		#
		# elif sensor == "matching":
		#     # frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
		#     template = frame[pedestrian_1_template_rect['y']:
		#                      pedestrian_1_template_rect['y'] + pedestrian_1_template_rect['h'],
		#                pedestrian_1_template_rect['x']:
		#                pedestrian_1_template_rect['x'] + pedestrian_1_template_rect['w']]
		#
		# if sensor == "hog":
		#     (rects, weights) = hog.detectMultiScale(ped_1_frame, winStride=(4, 4),
		#                                             padding=(8, 8), scale=1.05)
		#
		#     if len(weights) > 0:
		#         max_w_id = np.argmax(weights)
		#         z_x, z_y, z_w, z_h = rects[max_w_id]
		#
		#         z_x += z_w // 2
		#         z_y += z_h // 2
		#
		#         z_x += np.random.normal(0, NOISE_2['x'])
		#         z_y += np.random.normal(0, NOISE_2['y'])
		#
		# elif sensor == "matching":
		#     corr_map = cv2.matchTemplate(ped_1_frame, template, cv2.TM_SQDIFF)
		#     z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
		#
		#     z_w = pedestrian_1_template_rect['w']
		#     z_h = pedestrian_1_template_rect['h']
		#
		#     z_x += z_w // 2 + np.random.normal(0, NOISE_2['x'])
		#     z_y += z_h // 2 + np.random.normal(0, NOISE_2['y'])
		#
		# x, y = pedestrian_1_filter.process(z_x, z_y)
		#
		# out_frame = frame.copy()
		# cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
		# cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
		# cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
		#               (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
		#               (0, 0, 255), 2)
		#
		# cv2.imshow('Tracking', out_frame)
		# cv2.waitKey(1)
		# endregion
		
		# region Initialize Templates
		pedestrian_1_template = ped_1_frame[int(pedestrian_1_template_rect['y']):
											int(pedestrian_1_template_rect['y'] + pedestrian_1_template_rect['h']),
								int(pedestrian_1_template_rect['x']):
								int(pedestrian_1_template_rect['x'] + pedestrian_1_template_rect['w'])]
		pedestrian_2_template = ped_2_frame[int(pedestrian_2_template_rect['y']):
											int(pedestrian_2_template_rect['y'] + pedestrian_2_template_rect['h']),
								int(pedestrian_2_template_rect['x']):
								int(pedestrian_2_template_rect['x'] + pedestrian_2_template_rect['w'])]

		pedestrian_3_template = ped_3_frame[int(pedestrian_3_template_rect['y']):
											int(pedestrian_3_template_rect['y'] + pedestrian_3_template_rect['h']),
								int(pedestrian_3_template_rect['x']):
								int(pedestrian_3_template_rect['x'] + pedestrian_3_template_rect['w'])]
		# endregion

		if frame_num < 60:
			if pedestrian_1_filter is None and 1 in which_peds:
				pedestrian_1_filter = ps5.MDParticleFilter(frame=ped_1_frame,
													   template=pedestrian_1_template,
													   num_particles=1000,
													   sigma_exp=5,
													   sigma_dyn=10, template_coords=pedestrian_1_template_rect,
														   alpha=0.1, is_part_5=True)
				pedestrian_1_filter.process(ped_1_frame, resize_template=False)
				ped_1_frame = pedestrian_1_filter.render(ped_1_frame, rectangle_color=pedestrian_1_rect_color, pedestrian="1", is_part_5=True)

			elif 1 in which_peds:
				pedestrian_1_filter.process(ped_1_frame, resize_template=False)
				ped_1_frame = pedestrian_1_filter.render(ped_1_frame, rectangle_color=pedestrian_1_rect_color, pedestrian="1", is_part_5=True)
				# temp_frame = visualize_filter_pt_5(pf=pedestrian_1_filter, frame=ped_1_frame, pedestrian="1",
				#                                    show_weights=True, return_frame=True)

		if frame_num < 50:
			if pedestrian_2_filter is None and 2 in which_peds:
				pedestrian_2_filter = ps5.MDParticleFilter(frame=ped_2_frame,
													   template=pedestrian_2_template,
													   num_particles=1000,
													   sigma_exp=5,
													   sigma_dyn=10, template_coords=pedestrian_2_template_rect,
														   alpha=0.1, is_part_5=True, initial_momentum=-1)
				pedestrian_2_filter.process(ped_2_frame, resize_template=False)
				ped_2_frame = pedestrian_2_filter.render(ped_2_frame, rectangle_color=pedestrian_2_rect_color, pedestrian="2", is_part_5=True)
				# temp_frame = visualize_filter_pt_5(pf=pedestrian_2_filter, frame=ped_2_frame, pedestrian="2",
				#                                    show_weights=True, return_frame=True)
			elif 2 in which_peds:
				pedestrian_2_filter.process(ped_2_frame, resize_template=False)
				ped_2_frame = pedestrian_2_filter.render(ped_2_frame, rectangle_color=pedestrian_2_rect_color, pedestrian="2", is_part_5=True)
				# temp_frame = visualize_filter_pt_5(pf=pedestrian_2_filter, frame=ped_2_frame, pedestrian="2",
				# 								   show_weights=True, return_frame=True)

		if pedestrian_3_filter is None and 3 in which_peds and frame_num >= pedestrian_3_enter_frame:
			wait_val = 1
			pedestrian_3_filter = ps5.MDParticleFilter(frame=ped_3_frame,
												   template=pedestrian_3_template,
												   num_particles=1000,
												   sigma_exp=5,
												   sigma_dyn=10, template_coords=pedestrian_3_template_rect,
													   alpha=0.1, is_part_5=True, initial_momentum=1)
			pedestrian_3_filter.process(ped_3_frame, resize_template=False)
			ped_3_frame = pedestrian_3_filter.render(ped_3_frame, rectangle_color=pedestrian_3_rect_color, pedestrian="3", is_part_5=True)
		elif 3 in which_peds and frame_num >= pedestrian_3_enter_frame:
			pedestrian_3_filter.process(ped_3_frame, resize_template=False)
			ped_3_frame = pedestrian_3_filter.render(ped_3_frame, rectangle_color=pedestrian_3_rect_color, pedestrian="3", is_part_5=True)
			# temp_frame = visualize_filter_pt_5(pf=pedestrian_3_filter, frame=ped_3_frame, pedestrian="3",
			#                                    show_weights=True, return_frame=True)
		
		if 1 in which_peds:
			cv2.imshow('Pedestrian 1', ped_1_frame)
		if 2 in which_peds:
			cv2.imshow('Pedestrian 2', ped_2_frame)
		if 3 in which_peds:
			cv2.imshow("Pedestrian 3", ped_3_frame)
		cv2.waitKey(wait_val)

		# Render and save output, if indicated
		if frame_num in save_frames:
			frame_out = frame.copy()
			if 1 in which_peds and frame_num < 60:
				frame_out = pedestrian_1_filter.render(frame_out, rectangle_color=pedestrian_1_rect_color,
			                                         pedestrian="1", is_part_5=True)
			if 2 in which_peds and frame_num < 50:
				frame_out = pedestrian_2_filter.render(frame_out, rectangle_color=pedestrian_2_rect_color,
			                                         pedestrian="2", is_part_5=True)
			if 3 in which_peds and frame_num >= pedestrian_3_enter_frame:
				frame_out = pedestrian_3_filter.render(frame_out, rectangle_color=pedestrian_3_rect_color,
				                                       pedestrian="3", is_part_5=True)
			cv2.imwrite(save_frames[frame_num], frame_out)

		# Update frame number
		frame_num += 1
		if frame_num % 20 == 0:
			print('Working on frame {}'.format(frame_num))
	if save_video:
		video.release()
	cv2.destroyAllWindows()
	return 0


def part_6():
	"""Tracking pedestrians from a moving camera.

	Follow the instructions in the problem set instructions.

	Place all your work in this file and this section.
	"""
	save_video = False
	if save_video:
		width = 1330
		height = 480
		FPS = 10
		seconds = 10
		fourcc = cv2.VideoWriter_fourcc(*'MP42')
		video = cv2.VideoWriter('./pedestrian_tracker.mp4', fourcc, float(FPS), (width, height))
	save_frames = {
		60: os.path.join(output_dir, 'ps5-6-a-1.png'),
		160: os.path.join(output_dir, 'ps5-6-b-2.png'),
		186: os.path.join(output_dir, 'ps5-6-c-3.png'),
	}
	# Just pedestrian 1 face
	# pedestrian_1_template_rect = {'x': 94, 'y': 126, 'w': 33, 'h': 40}
	pedestrian_1_template_rect = {'x': 89, 'y': 32, 'w': 38, 'h': 72}
	wait_val = 1
	
	imgs_dir = os.path.join(input_dir, "follow")
	imgs_list = [f for f in os.listdir(imgs_dir)
	             if f[0] != '.' and f.endswith('.jpg')]
	imgs_list.sort()
	pedestrian_1_filter = None
	pedestrian_1_template = None
	pedestrian_1_rect_color = (40, 39, 214)
	which_peds = {1}
	
	# Initialize objects
	ksize = 9
	
	frame_num = 1
	for img in imgs_list:
		frame = cv2.imread(os.path.join(imgs_dir, img))
		ped_1_frame = np.copy(frame)
		# ped_1_frame = cv2.GaussianBlur(ped_1_frame, ksize=(ksize, ksize), sigmaX=2, sigmaY=2)
		ped_1_frame = cv2.filter2D(ped_1_frame, -1, kernel=np.ones((ksize,ksize))/ksize**2)
		
		# region Initialize Templates
		pedestrian_1_template = ped_1_frame[int(pedestrian_1_template_rect['y']):
		                                    int(pedestrian_1_template_rect['y'] + pedestrian_1_template_rect['h']),
		                        int(pedestrian_1_template_rect['x']):
		                        int(pedestrian_1_template_rect['x'] + pedestrian_1_template_rect['w'])]
		# endregion

		if pedestrian_1_filter is None and 1 in which_peds:
			pedestrian_1_filter = ps5.MDParticleFilter(frame=ped_1_frame,
			                                           template=pedestrian_1_template,
			                                           num_particles=1000,
			                                           sigma_exp=3,
			                                           sigma_dyn=6, template_coords=pedestrian_1_template_rect,
			                                           alpha=0.01, is_part_5=False, is_part_6=True)
			pedestrian_1_filter.process(ped_1_frame, resize_template=False)
			ped_1_frame = pedestrian_1_filter.render(ped_1_frame, rectangle_color=pedestrian_1_rect_color,
			                                         pedestrian="2", is_part_5=True, is_part_6=True)
		elif 1 in which_peds:
			if frame_num == 51 or frame_num == 104 or frame_num == 130 or frame_num == 145:
				pedestrian_1_filter.alpha = 0.9
			else:
				pedestrian_1_filter.alpha = 0.01
				
			pedestrian_1_filter.process(ped_1_frame, resize_template=False)
			ped_1_frame = pedestrian_1_filter.render(ped_1_frame, rectangle_color=pedestrian_1_rect_color,
			                                         pedestrian="2", is_part_5=True, is_part_6=True)
		temp_frame = visualize_filter_pt_5(pf=pedestrian_1_filter, frame=ped_1_frame, pedestrian="2",
		                                   show_weights=True, return_frame=False)
		
		if 1 in which_peds:
			cv2.imshow('Pedestrian 1', ped_1_frame)

		cv2.waitKey(wait_val)
		
		# Render and save output, if indicated
		if frame_num in save_frames:
			frame_out = frame.copy()
			frame_out = pedestrian_1_filter.render(frame_out, rectangle_color=pedestrian_1_rect_color,
			                                         pedestrian="2", is_part_5=True, is_part_6=True)
			cv2.imwrite(save_frames[frame_num], frame_out)
		
		# Update frame number
		frame_num += 1
		if frame_num % 20 == 0:
			print('Working on frame {}'.format(frame_num))
	if save_video:
		video.release()
	cv2.destroyAllWindows()
	return 0

if __name__ == '__main__':
	# part_1b()
	# part_1c()
	# part_2a()
	# part_2b()
	# part_3()
	# part_4()
	# part_5()
	part_6()
