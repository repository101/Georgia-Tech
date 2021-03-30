"""
CS6476: Problem Set 2 Experiment file

This script contains a series of function calls that run your ps2
implementation and output images so you can verify your results.
"""


import cv2

#import ps2
import ps2

# -- display utils
marker_color = (255, 0, 255)
text_color = (90, 90, 90)
text_thickness = 2
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5

def place_text(text, center, img, cache={}):
    if "y_offset" in cache:
        cache["y_offset"] *= -1
    else:
        cache["y_offset"] = -30
    size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    y = center[1] + cache["y_offset"]
    if size[0][0] + center[0] > img.shape[1]:
        x = center[0] - size[0][0] - 5
    else:
        x = center[0] + 5
    cv2.rectangle(img,(x,y - size[0][1] - size[1]),(x + size[0][0], y + size[0][1] - size[1]),(255,255,255),cv2.FILLED)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, text_thickness)




def draw_tl_center(image_in, center, state):
    """Marks the center of a traffic light image and adds coordinates
    with the state of the current image

    Use OpenCV drawing functions to place a marker that represents the
    traffic light center. Additionally, place text using OpenCV tools
    that show the numerical and string values of the traffic light
    center and state. Use the following format:

        ((x-coordinate, y-coordinate), 'color')

    See OpenCV's drawing functions:
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    Make sure the font size is large enough so that the text in the
    output image is legible.
    Args:
        image_in (numpy.array): input image.
        center (tuple): center numeric values.
        state (str): traffic light state values can be: 'red',
                     'yellow', 'green'.

    Returns:
        numpy.array: output image showing a marker representing the
        traffic light center and text that presents the numerical
        coordinates with the traffic light state.
    """
    center = (int(center[0]), int(center[1]))
    output = image_in
    cv2.drawMarker(output, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=11, thickness=2)
    text = "(({}, {}), '{}')".format(center[0], center[1], state)
    place_text(text, center, output)
    return output




def mark_traffic_signs(image_in, signs_dict):
    """Marks the center of a traffic sign and adds its coordinates.

    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}

    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', and 'warning'.

    Use cv2.putText to place the coordinate values in the output
    image.

    Args:
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.

    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """
    output = image_in
    items = []
    for k, center in signs_dict.items():
        items.append((int(center[0]), k, center))
    items.sort()

    for _, k, center in items:
        center = (int(center[0]), int(center[1]))
        cv2.drawMarker(output, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=11, thickness=2)
        text = "{}: ({}, {})".format(k, center[0], center[1])
        place_text(text, center, output)
    return output


def part_1():

    input_images = ['simple_tl', 'scene_tl_1', 'scene_tl_2', 'scene_tl_3']
    output_labels = ['ps2-1-a-1', 'ps2-1-a-2', 'ps2-1-a-3', 'ps2-1-a-4']

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_2():

    input_images = ['scene_dne_1', 'scene_stp_1', 'scene_constr_1',
                    'scene_wrng_1', 'scene_yld_1']

    output_labels = ['ps2-2-a-1', 'ps2-2-a-2', 'ps2-2-a-3', 'ps2-2-a-4',
                     'ps2-2-a-5']

    sign_fns = [ps2.ps2_2_a_1, ps2.ps2_2_a_2, ps2.ps2_2_a_3, ps2.ps2_2_a_4, ps2.ps2_2_a_5]

    sign_labels = ['no_entry', 'stop', 'construction', 'warning', 'yield']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_3():

    input_images = ['scene_some_signs', 'scene_all_signs']
    output_labels = ['ps2-3-a-1', 'ps2-3-a-2']

    sign_fns = [ps2.ps2_3_a_1, ps2.ps2_3_a_2]

    for img_in, label, fn in zip(input_images, output_labels, sign_fns):

        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(scene.copy())

        # print(coords)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_4():
    input_images = ['scene_some_signs_noisy', 'scene_all_signs_noisy']
    output_labels = ['ps2-4-a-1', 'ps2-4-a-2']

    sign_fns = [ps2.ps2_4_a_1, ps2.ps2_4_a_2]

    for img_in, label, fn in zip(input_images, output_labels, sign_fns):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(scene.copy())

        print(coords)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_5a():
    input_images = ['img-5-a-1', 'img-5-a-2', 'img-5-a-3']
    output_labels = ['ps2-5-a-1', 'ps2-5-a-2', 'ps2-5-a-3']
    sign_fns = [ps2.ps2_5_a_1, ps2.ps2_5_a_2, ps2.ps2_5_a_3]

    for img_in, label, fn in zip(input_images, output_labels, sign_fns):
        scene = cv2.imread("input_images/challenge_images/{}.png".format(img_in))
        coords = fn(scene.copy())

        # img_out = mark_traffic_signs(scene, coords)
        # cv2.imwrite("output/{}.png".format(label), img_out)


if __name__ == '__main__':
    print("part_1");part_1()
    print("part_2");part_2()
    print("part_3");part_3()
    print("part_4");part_4()
    print("part_5a");part_5a()
