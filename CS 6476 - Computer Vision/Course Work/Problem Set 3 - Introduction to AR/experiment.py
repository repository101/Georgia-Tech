"""
CS6476: Problem Set 3 Experiment file
This script consists in a series of function calls that run the ps3 
implementation and output images so you can verify your results.
"""

import os
import cv2
import numpy as np

import ps3


IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "./"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def helper_for_part_4_and_5(video_name, fps, frame_ids, output_prefix,
                            counter_init, is_part5):

    video = os.path.join(VID_DIR, video_name)
    image_gen = ps3.video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    if is_part5:
        advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
        src_points = ps3.get_corners_list(advert)

    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print("Processing fame {}".format(frame_num))

        markers = ps3.find_markers(image, template)

        if is_part5:
            homography = ps3.find_four_point_transform(src_points, markers)
            image = ps3.project_imageA_onto_imageB(advert, image, homography)

        else:
            
            for marker in markers:
                mark_location(image, marker)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()


def helper_for_part_7(input_video, input_image, output_video, output_prefix, frame_ids, fps, aruco_dict):

    # read a video
    vs = cv2.VideoCapture(os.path.join(VID_DIR, input_video))
    writer = None

    # read an image and find corner coordinates
    image = cv2.imread(os.path.join(IMG_DIR, input_image))
    src_points = ps3.get_corners_list(image)

    # iterate each frame of the video
    frame_num = 0
    output_counter = 0
    while True:
        frame_num += 1
        print("processing frame {}...".format(frame_num))

        # grab the next frame from the video
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        # detect all ArUco markers and their ID in the frame
        markers, ids = ps3.find_aruco_markers(frame, aruco_dict)
        if markers is not None and len(markers) == 4:
            # find the centers of four ArUco markers
            centers = ps3.find_aruco_center(markers)

            # project the input image onto a wall image with ArUco markers
            homography = ps3.find_four_point_transform(src_points, centers)
            frame = ps3.project_imageA_onto_imageB(image, frame, homography)

            # mark the center and place an ID string of each ArUco marker
            for center, id in zip(centers, ids):
                cv2.circle(frame, center, 4, (0, 255, 0), -1)
                cv2.putText(frame, "id=" + str(id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # check for video writer and if there is none, initialize one
        if writer is None:
            writer = mp4_video_writer(output_video, (frame.shape[1], frame.shape[0]), fps)

        # save frames according to the given frame ID list
        frame_id = frame_ids[output_counter % len(frame_ids)]
        if frame_num == frame_id:
            output_counter += 1
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, frame)

        # write the output frame to disk
        writer.write(frame)

    print("releasing the file pointers...")
    writer.release()
    vs.release()


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)


def part_1():

    print("\nPart 1:")

    input_images = ['sim_clear_scene.jpg', 'sim_noisy_scene_1.jpg',
                    'sim_noisy_scene_2.jpg']
    output_images = ['ps3-1-a-1.png', 'ps3-1-a-2.png', 'ps3-1-a-3.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        marker_positions = ps3.find_markers(scene, template)

        for marker in marker_positions:
            mark_location(scene, marker)

        save_image(img_out, scene)


def part_2():

    print("\nPart 2:")

    input_images = ['ps3-2-a_base.jpg', 'ps3-2-b_base.jpg',
                    'ps3-2-c_base.jpg', 'ps3-2-d_base.jpg', 'ps3-2-e_base.jpg']
    output_images = ['ps3-2-a-1.png', 'ps3-2-a-2.png', 'ps3-2-a-3.png',
                     'ps3-2-a-4.png', 'ps3-2-a-5.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = ps3.find_markers(scene, template)
        image_with_box = ps3.draw_box(scene, markers, 3)

        save_image(img_out, image_with_box)


def part_3():

    print("\nPart 3:")

    input_images = ['ps3-3-a_base.jpg', 'ps3-3-b_base.jpg', 'ps3-3-c_base.jpg']
    output_images = ['ps3-3-a-1.png', 'ps3-3-a-2.png', 'ps3-3-a-3.png']

    # Advertisement image
    advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    src_points = ps3.get_corners_list(advert)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):
        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = ps3.find_markers(scene, template)

        homography = ps3.find_four_point_transform(src_points, markers)

        projected_img = ps3.project_imageA_onto_imageB(advert, scene,
                                                       homography)

        save_image(img_out, projected_img)


def part_4_a():

    print("\nPart 4a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 1, False)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 4, False)


def part_4_b():

    print("\nPart 4b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 1, False)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 4, False)


def part_5_a():

    print("\nPart 5a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 1, True)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 4, True)


def part_5_b():

    print("\nPart 5b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 1, True)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 4, True)


def part_6():

    print("\nPart 6:")

    video_file = "ps3-4-a.mp4"
    my_video = "my-ad.mp4"  # Place your video in the input_video directory
    frame_ids = [355, 555, 725]
    fps = 40
    counter_init = 1

    # Todo: Complete this part on your own.
    video = os.path.join(VID_DIR, video_file)
    image_gen = ps3.video_frame_generator(video)
    ad_video = os.path.join(VID_DIR, my_video)
    ad_image_gen = ps3.video_frame_generator(ad_video)
    
    image = image_gen.__next__()
    ad_image = ad_image_gen.__next__()
    
    video_h, video_w, video_d = image.shape
    ad_h, ad_w, ad_d = ad_image.shape

    out_path = "ar_{}-{}".format("ps3-6-a"[4:], video_file)
    video_out = mp4_video_writer(out_path, (video_w, video_h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
    
    # This needs to be changed to image from video
    # advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    src_points = ps3.get_corners_list(ad_image)

    output_counter = counter_init

    frame_num = 1

    while image is not None:
    
        print("Processing fame {}".format(frame_num))
    
        markers = ps3.find_markers(image, template)
        homography = ps3.find_four_point_transform(src_points, markers)
        image = ps3.project_imageA_onto_imageB(ad_image, image, homography)

        frame_id = frame_ids[(output_counter - 1) % 3]
    
        if frame_num == frame_id:
            out_str = "ps3-6-a" + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1
    
        video_out.write(image)
    
        image = image_gen.__next__()
        ad_image = ad_image_gen.__next__()
        frame_num += 1

    video_out.release()


def part_7():

    print("\nPart 7:")

    input_video = "ps3-7.mp4"
    input_image = "my-image.png"
    output_video = "ps3-7-out.mp4"
    output_prefix = "ps3-7"
    frame_ids = [200, 300, 400]
    fps = 40
    aruco_dict = cv2.aruco.DICT_5X5_50  # do not change this

    helper_for_part_7(input_video, input_image, output_video, output_prefix, frame_ids, fps, aruco_dict)


if __name__ == '__main__':
    print("--- Problem Set 3 ---")
    # Comment out the sections you want to skip

    # part_1()
    # part_2()
    # part_3()
    # part_4_a()
    # part_4_b()
    # part_5_a()
    # part_5_b()
    # part_6()
    part_7()
