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
    
    pedestrian_1_template_rect = {'x': 95, 'y': 142, 'w': 50, 'h': 200}
    pedestrian_2_template_rect = {'x': 300, 'y': 200, 'w': 40, 'h': 100}
    wait_val = 1
    # pedestrian_2_template_rect = {'x': 280, 'y': 195, 'w': 50, 'h': 125}
    pedestrian_3_enter_frame = 28
    pedestrian_3_template_rect = {'x': 0, 'y': 170, 'w': 57, 'h': 247}
    
    # Define process and measurement arrays if you want to use other than the
    # default.
    num_particles = 1000
    sigma_exp = 5
    sigma_dyn = 10
    
    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()
    pedestrian_1_filter = None
    pedestrian_1_template = None
    pedestrian_1_rect_color = (0, 175, 0)
    pedestrian_2_filter = None
    pedestrian_2_template = None
    pedestrian_2_rect_color = (175, 0, 0)
    pedestrian_3_filter = None
    pedestrian_3_template = None
    pedestrian_3_rect_color = (0, 0, 175)

    # Initialize objects
    
    frame_num = 0
    for img in imgs_list:
        frame = cv2.imread(os.path.join(imgs_dir, img))
        ped_1_frame = np.copy(frame)
        ped_2_frame = np.copy(frame)
        ped_3_frame = np.copy(frame)
        
        if pedestrian_1_template is None:
            pedestrian_1_template = frame[int(pedestrian_1_template_rect['y']):
                             int(pedestrian_1_template_rect['y'] + pedestrian_1_template_rect['h']),
                             int(pedestrian_1_template_rect['x']):
                             int(pedestrian_1_template_rect['x'] + pedestrian_1_template_rect['w'])]

        if pedestrian_1_filter is None:
            pedestrian_1_filter = ps5.MDParticleFilter(frame=ped_1_frame,
                                                   template=pedestrian_1_template,
                                                   num_particles=num_particles,
                                                   sigma_exp=sigma_exp,
                                                   sigma_dyn=sigma_dyn, template_coords=pedestrian_1_template_rect,
                                                       alpha=0.1, is_part_5=True)
        
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


        # if pedestrian_2_template is None:
        #     pedestrian_2_template = frame[int(pedestrian_2_template_rect['y']):
        #                      int(pedestrian_2_template_rect['y'] + pedestrian_2_template_rect['h']),
        #                      int(pedestrian_2_template_rect['x']):
        #                      int(pedestrian_2_template_rect['x'] + pedestrian_2_template_rect['w'])]
        #
        # if pedestrian_2_filter is None:
        #     pedestrian_2_filter = ps5.MDParticleFilter(frame=ped_2_frame,
        #                                            template=pedestrian_2_template,
        #                                            num_particles=num_particles,
        #                                            sigma_exp=sigma_exp,
        #                                            sigma_dyn=15, template_coords=pedestrian_2_template_rect,
        #                                                alpha=0.1, is_part_5=True, initial_momentum=-1)

        if pedestrian_3_template is None and pedestrian_3_filter is None and frame_num >= pedestrian_3_enter_frame:
            if pedestrian_3_template is None:
                pedestrian_3_template = frame[int(pedestrian_3_template_rect['y']):
                                 int(pedestrian_3_template_rect['y'] + pedestrian_3_template_rect['h']),
                                 int(pedestrian_3_template_rect['x']):
                                 int(pedestrian_3_template_rect['x'] + pedestrian_3_template_rect['w'])]

            if pedestrian_3_filter is None:
                pedestrian_3_filter = ps5.MDParticleFilter(frame=ped_3_frame,
                                                       template=pedestrian_3_template,
                                                       num_particles=num_particles,
                                                       sigma_exp=sigma_exp,
                                                       sigma_dyn=15, template_coords=pedestrian_3_template_rect,
                                                           alpha=0.1, is_part_5=True, initial_momentum=1)

        # Process frame
        pedestrian_1_filter.process(ped_1_frame, resize_template=False)
        # pedestrian_2_filter.process(ped_2_frame, resize_template=False)
        if pedestrian_3_filter is not None:
            pedestrian_3_filter.process(ped_3_frame, resize_template=False)

        # out_frame = frame.copy()
        # pf.render(out_frame)
        # video.write(out_frame)
        # pedestrian_1_filter.show_weights(wait=0)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pedestrian_1_filter.render(out_frame, rectangle_color=pedestrian_1_rect_color)
            # pedestrian_2_filter.render(out_frame, rectangle_color=pedestrian_2_rect_color)
            # pedestrian_1_filter.show_weights(wait=0)
            temp_frame = 0
            # temp_frame = visualize_filter_pt_5(pf=pedestrian_1_filter, frame=out_frame, pedestrian="1", show_weights=True, return_frame=True)
            # temp_frame = visualize_filter_pt_5(pf=pedestrian_2_filter, frame=out_frame, pedestrian="2", show_weights=True, return_frame=True)
            # pedestrian_2_filter.render(out_frame, rectangle_color=pedestrian_2_rect_color, is_pt_5=True)
            # visualize_filter_pt_5(pf=pedestrian_2_filter, frame=out_frame, pedestrian="2")
            
            if pedestrian_3_filter is not None:
                pedestrian_3_filter.render(out_frame, rectangle_color=pedestrian_3_rect_color)
                visualize_filter_pt_5(pf=pedestrian_3_filter, frame=out_frame, pedestrian="3")

            if save_video:
                video.write(temp_frame)
            cv2.imshow('Tracking', out_frame)
            if save_video:
                cv2.waitKey(1)
            else:
                cv2.waitKey(wait_val)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            # pedestrian_1_filter.render(frame_out, rectangle_color=pedestrian_1_rect_color, is_pt_5=True)
            # pedestrian_2_filter.render(frame_out, rectangle_color=pedestrian_2_rect_color, is_pt_5=True)
            # if pedestrian_3_filter is not None:
            #     pedestrian_3_filter.render(frame_out, rectangle_color=pedestrian_3_rect_color, is_pt_5=True)
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
    raise NotImplementedError


if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    part_5()
    # part_6()
