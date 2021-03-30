"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


# Globals found in experiment.py
marker_color = (255, 0, 255)
text_color = (90, 90, 90)
text_thickness = 3
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3


class Traffic_Light:
    def __init__(self, points, color_image, hsv_image, gray_image):
        self.red_light_location = None
        self.green_light_location = None
        self.yellow_light_location = None
        self.light_currently_lit = None
        self.light_currently_lit_as_string = None
        self.green_light_votes = []
        self.red_light_votes = []
        self.yellow_light_votes = []
        self.which_light_is_lit_votes = []
        self.all_points = points
        self.point_patch = {}
        self.alignment = None
        self.color_image = color_image
        self.hsv_image = hsv_image
        self.grayscale_image = gray_image
        self.determine_point_colors()

    def determine_point_colors(self):
        color_patches = np.zeros(shape=(len(self.all_points),), dtype=object)
        hsv_patches = np.zeros(shape=(len(self.all_points),), dtype=object)
        gray_patches = np.zeros(shape=(len(self.all_points),))
        count = 0
        for point in self.all_points:
            x_range = (int(point[0] - 10), int(point[0] + 10))
            y_range = (int(point[1] - 10), int(point[1] + 10))

            color_segment = self.color_image[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
            hsv_segment = self.hsv_image[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
            gray_segment = self.grayscale_image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            self.point_patch[f"({point[0]},{point[1]})"] = {(point[0], point[1]): {"Color Patch": color_segment,
                                                                                   "HSV Patch": hsv_segment,
                                                                                   "Gray Patch": gray_segment}}
            color_patches[count] = (
                np.mean(color_segment[:, :, 0]), np.mean(color_segment[:, :, 1]), np.mean(color_segment[:, :, 2]))
            hsv_patches[count] = (
                np.mean(hsv_segment[:, :, 0]), np.mean(hsv_segment[:, :, 1]), np.mean(hsv_segment[:, :, 2]))
            gray_patches[count] = np.mean(gray_segment)
            # Range for lower red
            low_red_mask, high_red_mask = create_red_hsv_mask(hsv_segment)
            red_mask = low_red_mask + high_red_mask
            green_mask = create_green_hsv_mask(hsv_segment)
            temp_red = cv2.bitwise_and(hsv_segment, hsv_segment, mask=red_mask)
            temp_green = cv2.bitwise_and(hsv_segment, hsv_segment, mask=green_mask)
            if ~np.all(temp_red == 0):
                # We have a red segment
                self.red_light_location = point
            elif ~np.all(temp_green == 0):
                # We have a red segment
                self.green_light_location = point
            else:
                self.yellow_light_location = point

            count += 1
        gray_currently_lite_up = None
        if gray_patches[1] > 200:
            # Vote Yellow
            gray_currently_lite_up = 1
            self.light_currently_lit = self.all_points[1]
            self.light_currently_lit_as_string = "yellow"
        elif gray_patches[1] < 200 and gray_patches[2] > 100:
            # Vote Green
            gray_currently_lite_up = 2
            self.light_currently_lit = self.all_points[2]
            self.light_currently_lit_as_string = "green"
        elif gray_patches[1] < 200 and gray_patches[2] < 100:
            # Vote Red
            gray_currently_lite_up = 0
            self.light_currently_lit = self.all_points[0]
            self.light_currently_lit_as_string = "red"
        self.which_light_is_lit_votes.append(gray_currently_lite_up)


def find_lines(edges):
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 9, None, 7, 1)


def place_text(text, center, img, cache={}):
    # Provided in experiment.py
    if "y_offset" in cache:
        cache["y_offset"] *= -1
    else:
        cache["y_offset"] = -30
    size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    y = center[1] + cache["y_offset"] - 50
    if size[0][0] + center[0] > img.shape[1]:
        x = center[0] - size[0][0] + 20
    else:
        x = center[0] + 50
    cv2.rectangle(img, (x, y - size[0][1] - size[1]), (x + size[0][0], y + size[0][1] - size[1]), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, text_thickness)


def calc_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calc_slope(pt1, pt2):
    return abs(pt1[0] - pt2[0]) / abs(pt1[1] - pt2[1])


def calc_angle_in_degrees(pt1, pt2):
    # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
    return np.rad2deg(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))


def create_yellow_hsv_mask(hsv_img, is_challenge=False):
    lower = np.array([20, 40, 150])
    upper = np.array([40, 255, 255])
    if is_challenge:
        lower = np.array([20, 40, 150])
        upper = np.array([40, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def create_red_hsv_mask(hsv_img):
    lower_lower_red = np.array([0, 120, 70])
    lower_upper_red = np.array([10, 255, 255])
    upper_lower_red = np.array([170, 120, 70])
    upper_upper_red = np.array([180, 255, 255])
    return cv2.inRange(hsv_img, lower_lower_red, lower_upper_red), cv2.inRange(hsv_img, upper_lower_red,
                                                                               upper_upper_red)


def create_green_hsv_mask(hsv_img):
    lower = np.array([35, 25, 25])
    upper = np.array([70, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def create_white_hsv_mask(hsv_img):
    lower = np.array([0, 0, 230])
    upper = np.array([180, 35, 255])
    return cv2.inRange(hsv_img, lower, upper)


def create_blue_hsv_mask(hsv_img):
    lower = np.array([100, 40, 65])
    upper = np.array([140, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def create_orange_hsv_mask(hsv_img):
    lower = np.array([10, 100, 20])
    upper = np.array([25, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def create_black_hsv_mask(hsv_img):
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 65])
    return cv2.inRange(hsv_img, lower, upper)


def find_triangle_extrema(x_cords, y_cords, img):
    # determine if extema on plane
    temp_x = np.unique(x_cords, return_counts=True)
    temp_y = np.unique(y_cords, return_counts=True)
    max_x_cord = x_cords.max(), temp_x[1][temp_x[0] == x_cords.max()][0]
    min_x_cord = x_cords.min(), temp_x[1][temp_x[0] == x_cords.min()][0]
    max_y_cord = y_cords.max(), temp_y[1][temp_y[0] == y_cords.max()][0]
    min_y_cord = y_cords.min(), temp_y[1][temp_y[0] == y_cords.min()][0]
    max_count = max([max_x_cord[1], min_x_cord[1], max_y_cord[1], min_y_cord[1]])
    if max_x_cord[1] == max_count:
        # Plane is located here and two extrema will have this max X cord
        locations_at_plane = y_cords[x_cords == max_x_cord[0]]
        point_1 = (max_x_cord[0], locations_at_plane.min())
        point_2 = (max_x_cord[0], locations_at_plane.max())
        point_3 = (x_cords.min(), y_cords[x_cords == x_cords.min()][0])
        return point_1, point_2, point_3
    elif min_x_cord[1] == max_count:
        # Plane is located here and two extrema will have this min X cord
        locations_at_plane = y_cords[x_cords == min_x_cord[0]]
        point_1 = (min_x_cord[0], locations_at_plane.min())
        point_2 = (min_x_cord[0], locations_at_plane.max())
        point_3 = (x_cords.max(), y_cords[x_cords == x_cords.max()][0])
        return point_1, point_2, point_3
    elif max_y_cord[1] == max_count:
        # Plane is located here and two extrema will have this max Y cord
        locations_at_plane = x_cords[y_cords == max_y_cord[0]]
        point_1 = (locations_at_plane.min(), max_y_cord[0])
        point_2 = (locations_at_plane.max(), max_y_cord[0])
        # Must be angled upward as max_y_cord is the plane
        point_3 = (x_cords[y_cords == y_cords.min()][0], y_cords.min())

        return point_1, point_2, point_3
    elif min_y_cord[1] == max_count:
        # Plane is located here and two extrema will have this min Y cord
        locations_at_plane = x_cords[y_cords == min_y_cord[0]]
        point_1 = (locations_at_plane.min(), min_y_cord[0])
        point_2 = (locations_at_plane.max(), min_y_cord[0])
        # Must be angled downward as min_y_cord is the plane
        point_3 = (x_cords[y_cords == y_cords.max()][0], y_cords.max())
        # out_img = cv2.cvtColor(img, cv2.C)
        return point_1, point_2, point_3
    else:
        return (max_x_cord[0], max_y_cord[1]), (min_x_cord[1], max_y_cord[1]), (min_x_cord[1], min_y_cord[1])


def setup(input_image):
    temp = cv2.bilateralFilter(input_image, 9, 75, 75)
    return cv2.bilateralFilter(input_image, 9, 75, 75), cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
    x_intersect = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y_intersect = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return (x_intersect, y_intersect)


def group_and_remove_radi(radi_results, circles):
    new_results_values = np.copy(radi_results[0])
    new_results_counts = np.copy(radi_results[1])
    group_count = 0
    group_tracker = {}
    visited = set()
    for i in range(radi_results[0].shape[0]):
        if radi_results[0][i] not in visited:
            t = np.isclose(radi_results[0][i], radi_results[0][i + 1:], atol=0.3)
            visited.add(radi_results[0][i])
            b = radi_results[0][i + 1:][t]
            if len(b) > 0:
                group_count += 1
                temp_group = np.append(radi_results[0][i], b)
                for m in temp_group:
                    visited.add(m)
                group_tracker[f"{group_count}"] = np.append(radi_results[0][i], b)
                avg_radi = np.mean(group_tracker[f"{group_count}"])
                circles[0][:, 2][np.in1d(circles[0][:, 2], temp_group)] = avg_radi
                total_count = radi_results[1][i] + np.sum(radi_results[1][i + 1:][t])
                mask = ~np.in1d(new_results_values, temp_group)
                new_results_values = new_results_values[mask]
                new_results_counts = new_results_counts[mask]
                new_results_values = np.append(new_results_values, avg_radi)
                new_results_counts = np.append(new_results_counts, total_count)

    temp_mask = new_results_counts > 1
    return new_results_values[temp_mask], new_results_counts[temp_mask], circles


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    gray_scale_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    circles = cv2.HoughCircles(gray_scale_img, cv2.HOUGH_GRADIENT, dp=1, minDist=radii_range[-1] + 1, param1=20,
                               param2=10,
                               minRadius=radii_range[0], maxRadius=radii_range[-1] + 2)

    temp_values = np.unique(circles[0][:, 2], return_counts=True)
    radi_values, radi_count, circles = group_and_remove_radi(temp_values, circles=circles)
    groups = []
    for i in range(circles.shape[1]):
        current_circle = circles[0][i]
        # all_circles_with_the_same_radius_as_current_circle = circles[0][circles[0]]
        close_mask = np.isclose(circles[0][:, 2], current_circle[2], atol=0.3)
        all_circles_with_same_radius = circles[0][close_mask]
        distance = np.sqrt(((all_circles_with_same_radius[:, 0] - current_circle[0]) ** 2 + (
                all_circles_with_same_radius[:, 1] - current_circle[1]) ** 2))
        threshold = 3 * current_circle[2]
        threshold_mask = distance < threshold
        close_circles = all_circles_with_same_radius[threshold_mask]
        if len(close_circles) > 1:
            groups.append(close_circles)

    traffic_lights = []
    for group in groups:
        # Validate group
        if len(group) == 3:
            distance_a_b = np.sqrt((group[0][0] - group[1][0]) ** 2 + (group[0][1] - group[1][1]) ** 2)
            distance_b_c = np.sqrt((group[1][0] - group[2][0]) ** 2 + (group[1][1] - group[2][1]) ** 2)
            distance_c_a = np.sqrt((group[2][0] - group[0][0]) ** 2 + (group[2][1] - group[0][1]) ** 2)
            if np.isclose(distance_a_b, distance_b_c, atol=0.3):
                # check distance c_a to be ~2x distance_a or b
                if np.isclose(distance_c_a, 2 * distance_a_b, atol=0.3):
                    # Create a Light group
                    traffic_lights.append(Traffic_Light(points=group, color_image=img_in,
                                                        hsv_image=hsv_img, gray_image=gray_scale_img))
            elif np.isclose(distance_b_c, distance_c_a, atol=0.3):
                # Check distance between
                if np.isclose(distance_a_b, 2 * distance_c_a, atol=0.3):
                    # Create a Light group
                    traffic_lights.append(Traffic_Light(points=group, color_image=img_in,
                                                        hsv_image=hsv_img, gray_image=gray_scale_img))
        else:
            pass

    return (traffic_lights[0].yellow_light_location[0], traffic_lights[0].yellow_light_location[1]), traffic_lights[
        0].light_currently_lit_as_string


def find_line_intersections(line_1, line_2):
    result = np.linalg.solve(line_1, line_2)
    return result


def find_triangles(lines):
    lines = np.round(lines, 3)
    sides = []
    for line in range(len(lines)):
        current_theta = lines[line][0][1]
        current_theta_in_degrees = current_theta * (180 / np.pi)
        all_theta_in_degrees = lines[:][:, 0][:, 1] * (180 / np.pi)
        line_delta = all_theta_in_degrees - current_theta_in_degrees

    verticies = []
    for j in range(3):
        verticies.append(find_line_intersections(sides[j - 1], sides[j]))
    return verticies


def yield_sign_detection(img_in, robust=False):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img_in = cv2.bilateralFilter(img_in, 9, 75, 75)
    # Sign primarily white with edges being red, Triangular in shape
    colors_found_in_sign = ["Red", "White"]
    hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    red_mask_low, red_mask_high = create_red_hsv_mask(hsv_image)
    white_mask = create_white_hsv_mask(hsv_image)
    red_mask = red_mask_low + red_mask_high
    filtered_image_white = cv2.bitwise_and(img_in, img_in, mask=white_mask)
    filtered_image_red = cv2.bitwise_and(img_in, img_in, mask=red_mask)
    if np.count_nonzero(white_mask > 0) > (white_mask.size * 0.5):
        gray_filtered = cv2.cvtColor(filtered_image_red, cv2.COLOR_BGR2GRAY)
    # Do not combine white and red
    else:
        filtered_image = cv2.bitwise_and(img_in, img_in, mask=white_mask + red_mask)
        gray_filtered = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        if robust:
            gray_filtered[gray_filtered > 0] = 255
    gray_and_white_pixels = np.where(gray_filtered > 0)
    pixels_x = gray_and_white_pixels[1]
    pixels_y = gray_and_white_pixels[0]
    extrema = find_triangle_extrema(pixels_x, pixels_y, img_in)
    line_one = (extrema[0], extrema[1])
    line_one_center_point = (((line_one[0][0] + line_one[1][0]) // 2), ((line_one[0][1] + line_one[1][1]) // 2))
    line_two = (extrema[1], extrema[2])
    line_two_center_point = (((line_two[0][0] + line_two[1][0]) // 2), ((line_two[0][1] + line_two[1][1]) // 2))
    line_three = (extrema[2], extrema[0])
    line_three_center_point = (
        ((line_three[0][0] + line_three[1][0]) // 2), ((line_three[0][1] + line_three[1][1]) // 2))
    non_robust_center = (
        (extrema[0][0] + extrema[1][0] + extrema[2][0]) // 3, (extrema[0][1] + extrema[1][1] + extrema[2][1]) // 3)
    if not robust:
        return non_robust_center
    else:
        filtered_image_combined = cv2.bitwise_or(filtered_image_white, filtered_image_red, mask=red_mask + white_mask)
        gray_filtered = cv2.cvtColor(filtered_image_combined, cv2.COLOR_BGR2GRAY)
        gray_filtered[gray_filtered > 0] = 255
        temp_gray = np.copy(gray_filtered)
        temp_gray[temp_gray > 0] = 255
        edges = cv2.Canny(temp_gray, 0, 0, None, 3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=56,
                                minLineLength=6,
                                maxLineGap=7)
        if lines is not None:
            lines = lines.reshape(lines.shape[0], -1)
            min_length_lines = [l for l in lines if calc_distance((l[0], l[1]), (l[2], l[3])) > 5]
            acceptable_angles_in_degrees = [28, 29, 30, 31, 32, 58, 59, 60, 61, 62]
            acceptable_angles_in_radians = [(i * (np.pi / 180)) for i in acceptable_angles_in_degrees]
            temp = {(i[0], i[1], i[2], i[3]): np.round(np.abs(calc_angle_in_degrees((i[0], i[1]), (i[2], i[3]))))
                    for i in
                    min_length_lines}
            filtered_lines = []
            for key, val in temp.items():
                if 0 < val < 90 and val != 45 and val in acceptable_angles_in_degrees:
                    filtered_lines.append(key)
            filtered_lines = np.asarray(filtered_lines)
            if len(filtered_lines) > 0:
                line_stack = np.float32(np.vstack((filtered_lines[:, [0, 1]], filtered_lines[:, [2, 3]])))
                for i in range(len(line_stack)):
                    current_line = line_stack[i]
                    if i + 1 < len(line_stack):
                        next_line = line_stack[i + 1]
                        if np.abs(current_line[0] - next_line[0]) < 5:
                            if np.abs(current_line[1] - next_line[1]) < 5:
                                # merge those points
                                new_point = (
                                    (current_line[0] + next_line[0]) // 2, (current_line[1] + next_line[1]) // 2)
                                temp = line_stack[0:i, :]
                                temp2 = line_stack[i + 2:, :]
                                line_stack = np.concatenate((temp, temp2, np.asarray([[new_point[0], new_point[1]]])))
                                break
                if len(line_stack) <= 3:
                    new_cords = (int(np.mean(line_stack[:, 0])), int(np.mean(line_stack[:, 1])))
                    return new_cords

                else:
                    max_y = line_stack[:, 1].max()
                    min_y = line_stack[:, 1].min()
                    x_associated_with_max_y = line_stack[:, 0][line_stack[:, 1] == max_y]
                    x_associated_with_min_y = line_stack[:, 0][line_stack[:, 1] == min_y]
                    if len(x_associated_with_max_y) > 1 and len(x_associated_with_min_y) > 1:
                        if np.abs(x_associated_with_max_y[0] - x_associated_with_max_y[1]) < 5:
                            new_point = ((x_associated_with_max_y[0] + x_associated_with_max_y[1]) // 2, (max_y))
                            new_lines = np.vstack((line_stack[:, :][line_stack[:, 1] != max_y], new_point))
                            cords = int(np.sum(new_lines[:, 0]) // 3), int(np.sum(new_lines[:, 1]) // 3)
                            return cords
                        if np.abs(x_associated_with_min_y[0] - x_associated_with_min_y[1]) < 5:
                            new_point = ((x_associated_with_min_y[0] + x_associated_with_min_y[1]) // 2, (min_y))
                            new_lines = np.vstack((line_stack[:, :][line_stack[:, 1] != min_y], new_point))
                            cords = int(np.sum(new_lines[:, 0]) // 3), int(np.sum(new_lines[:, 1]) // 3)
                            return cords
        else:
            return None


def stop_sign_detection(img_in, robust=False):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    # Sign primarily red with white letters, octagonal in shape
    colors_found_in_sign = ["Red", "White"]
    img_in, hsv_image = setup(img_in)
    red_mask_low, red_mask_high = create_red_hsv_mask(hsv_image)
    red_mask = red_mask_low + red_mask_high
    white_mask = create_white_hsv_mask(hsv_image)
    filtered_image_red = cv2.bitwise_and(img_in, img_in, mask=red_mask)
    filtered_image_white = cv2.bitwise_and(img_in, img_in, mask=white_mask)
    a = np.count_nonzero(white_mask > 0)
    if a > (white_mask.size * 0.5):
        # Dont combine filter
        gray_img = cv2.cvtColor(filtered_image_red, cv2.COLOR_BGR2GRAY)
        gray_img[gray_img > 0] = 255
    else:
        combined_red_and_white = cv2.bitwise_or(filtered_image_red, filtered_image_white)
        gray_img = cv2.cvtColor(combined_red_and_white, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.bilateralFilter(gray_img, 3, 75, 75)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 1)
    edges = cv2.Canny(gray_img, 0, 0, None, 3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=32, minLineLength=10,
                            maxLineGap=2)
    pixels = np.where(gray_img > 0)
    min_x = pixels[0].min()
    max_x = pixels[0].max()
    min_y = pixels[1].min()
    max_y = pixels[1].max()
    non_robust_center = ((min_y + max_y) // 2, (min_x + max_x) // 2)

    if not robust:
        return non_robust_center
    else:

        temp_gray = np.copy(gray_img)
        temp_gray[temp_gray > 0] = 255
        edges = cv2.Canny(temp_gray, 0, 0, None, 3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15,
                                minLineLength=8,
                                maxLineGap=8)
        if lines is not None:
            lines = lines.reshape(lines.shape[0], -1)
            min_length_lines = [l for l in lines if calc_distance((l[0], l[1]), (l[2], l[3])) > 5]
            acceptable_angles_in_degrees = [44, 45, 46, 134, 135, 136, 224, 225, 226, 314, 315, 316]
            acceptable_angles_in_radians = [(i * (np.pi / 180)) for i in acceptable_angles_in_degrees]
            temp = {(i[0], i[1], i[2], i[3]): np.round(np.abs(calc_angle_in_degrees((i[0], i[1]), (i[2], i[3])))) for i
                    in
                    min_length_lines}
            filtered_lines = []
            for key, val in temp.items():
                if val == 90 or val in acceptable_angles_in_degrees:
                    filtered_lines.append(key)
            all_x = lines[:8, [0, 2]]
            all_y = lines[:8, [1, 3]]
            point = int(np.mean(all_x)), int(np.mean(all_y))
            filtered_lines = np.asarray(filtered_lines)

            test = np.float32(np.vstack((filtered_lines[:, [0, 1]], filtered_lines[:, [2, 3]])))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, label, center = cv2.kmeans(test, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            if center is not None:
                return center[1][0], center[1][1]


def warning_sign_detection(img_in, robust=False, is_challenge=False):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    if not is_challenge:
        # Sign primarily yellow with thin black line near edges
        colors_found_in_sign = ["Yellow", "Black"]
        # Median Filter to remove black line
        img_in = cv2.medianBlur(img_in, 3)
        img_in = cv2.dilate(img_in, np.ones((3)))
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        yellow_mask = create_yellow_hsv_mask(hsv_image)
        filtered_image_yellow = cv2.bitwise_and(img_in, img_in, mask=yellow_mask)
        yellow_count = np.count_nonzero(yellow_mask > 0)
        if robust and yellow_count == 0:
            return None

        gray_filtered = cv2.cvtColor(filtered_image_yellow, cv2.COLOR_BGR2GRAY)
        gray_filtered[gray_filtered > 0] = 255
        edges = cv2.Canny(gray_filtered, 0, 0, apertureSize=3)
        gray_img = cv2.cvtColor(filtered_image_yellow, cv2.COLOR_BGR2GRAY)
        gray_img[gray_img > 0] = 255
        yellow_pixels = np.where(gray_img > 0)
        if robust:
            yellow_count = np.count_nonzero(gray_img > 0)
            if yellow_count == 0:
                return None
        test = np.float32(np.vstack(yellow_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(test.T, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if not robust:
            circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1,
                                       minDist=20, param1=28,
                                       param2=13,
                                       minRadius=5, maxRadius=40)
            if circles is not None:
                if len(circles) > 0:
                    usable_centers = []
                    if center is not None:
                        for i in center:
                            for circle in circles[0, :]:
                                circle_center = (circle[0], circle[1])
                                radius = circle[2]
                                dst = np.sqrt(((circle_center[0] - i[1]) ** 2 + (circle_center[1] - i[0]) ** 2))
                                if dst > radius:
                                    usable_centers.append((np.round(i[1]), np.round(i[0])))

                        return usable_centers[0][0], usable_centers[0][1]
            else:
                return int(center[0][1]), int(center[0][0])
        else:
            test = test.T
            min_x = test[:, 0].min()
            max_x = test[:, 0].max()
            min_y = test[:, 1].min()
            max_y = test[:, 1].max()
            non_robust_cords = (int((max_y + min_y) // 2), int((max_x + min_x) // 2))
            if not robust:
                return non_robust_cords
            else:
                gray_filtered[gray_filtered > 0] = 255
                temp_gray = np.copy(gray_filtered)
                temp_gray[temp_gray > 0] = 255

                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=19,
                                        minLineLength=15,
                                        maxLineGap=9)
                if lines is not None:
                    lines = lines.reshape(lines.shape[0], -1)
                    min_length_lines = [l for l in lines if calc_distance((l[0], l[1]), (l[2], l[3])) > 5]
                    acceptable_angles_in_degrees = [43, 44, 45, 46, 47]
                    acceptable_angles_in_radians = [(i * (np.pi / 180)) for i in acceptable_angles_in_degrees]
                    temp = {
                        (i[0], i[1], i[2], i[3]): np.round(np.abs(calc_angle_in_degrees((i[0], i[1]), (i[2], i[3]))))
                        for i in
                        min_length_lines}
                    filtered_lines = []
                    for key, val in temp.items():
                        if val in acceptable_angles_in_degrees:
                            filtered_lines.append(key)
                    filtered_lines = np.asarray(filtered_lines)

                    test = np.float32(np.vstack((filtered_lines[:, [0, 1]], filtered_lines[:, [2, 3]])))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, label, center_cords = cv2.kmeans(test, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    min_x = test[:, 0].min()
                    max_x = test[:, 0].max()
                    min_y = test[:, 1].min()
                    max_y = test[:, 1].max()
                    x_cord = int((min_x + max_x) // 2)
                    y_cord = int((min_y + max_y) // 2)
                    return x_cord, y_cord
    else:
        return


def construction_sign_detection(img_in, robust=False, is_challenge=False):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # Primarily Orange with a dark red line near edge. Diamond in shape
    colors_found_in_sign = ["Orange", "Black"]
    # Median Filter to remove black line
    img_in = cv2.medianBlur(img_in, 3)
    img_in = cv2.dilate(img_in, np.ones((3)))
    hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    orange_mask = create_orange_hsv_mask(hsv_image)
    filtered_image_orange = cv2.bitwise_and(img_in, img_in, mask=orange_mask)
    gray_filtered = cv2.cvtColor(filtered_image_orange, cv2.COLOR_BGR2GRAY)
    gray_filtered[gray_filtered > 0] = 255
    orange_pixels = np.where(gray_filtered > 0)
    if len(orange_pixels[0]) > 0:
        non_robust_center = ((orange_pixels[1].min() + orange_pixels[1].max()) // 2,
                             (orange_pixels[0].min() + orange_pixels[0].max()) // 2)
    else:
        return None
    if not robust:
        return non_robust_center
    else:
        test = np.float32(np.vstack((orange_pixels[0], orange_pixels[1]))).T
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(test, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if center is not None:
            if len(center) > 0:
                center = center[0]
            return center[1], center[0]
        else:
            return None


def do_not_enter_sign_detection(img_in, robust=False, is_challenge=False):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    if not is_challenge:
        img_in = cv2.bilateralFilter(img_in, 3, 75, 75)
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        colors_found_in_sign = ["Red", "White"]
        red_mask_low, red_mask_high = create_red_hsv_mask(hsv_image)
        red_mask = red_mask_low + red_mask_high
        white_mask = create_white_hsv_mask(hsv_image)

        filtered_image_red = cv2.bitwise_and(img_in, img_in, mask=red_mask)
        filtered_image_white = cv2.bitwise_and(img_in, img_in, mask=white_mask)
        if np.count_nonzero(white_mask > 0) > (white_mask.size * 0.5):
            gray_t = cv2.cvtColor(filtered_image_red, cv2.COLOR_BGR2GRAY)
            gray_t[gray_t > 0] = 255
        else:
            combined = cv2.bitwise_or(filtered_image_white, filtered_image_red)
            gray_t = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray_t, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=10, param1=15,
                                   param2=10,
                                   minRadius=5, maxRadius=40)

        dist_queue = {}
        distant_tracker = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                centers_of_all_other_circles = circles[0, :][:, 0:2]
                dist = np.mean(np.sqrt(((i[0] - centers_of_all_other_circles[:, 0]) ** 2 +
                                        (i[1] - centers_of_all_other_circles[:, 1]) ** 2)))
                dist_queue[dist] = (i[0], i[1], i[2])
                distant_tracker.append(dist)

        # Calculate distance between white points and circle found
        cords_for_all_white_points = np.where(white_mask == 255)
        if len(circles) > 0:
            circles = np.ravel(circles)
            dist = np.sqrt(
                (circles[0] - cords_for_all_white_points[0]) ** 2 + (circles[1] - cords_for_all_white_points[1]) ** 2)
            filtered_dist = dist < circles[2]
            valid_y = cords_for_all_white_points[0][filtered_dist]
            valid_x = cords_for_all_white_points[1][filtered_dist]
            tempImg = np.copy(gray_t)
            if len(valid_x) > 0:
                tempImg[valid_x, valid_y] = 255
                new_img = cv2.bitwise_and(tempImg, white_mask)
                white_within_sign = np.where(new_img > 0)
                min_white_x = white_within_sign[0].min()
                max_white_x = white_within_sign[0].max()
                min_white_y = white_within_sign[1].min()
                max_white_y = white_within_sign[1].max()
                x_cord = (min_white_x + max_white_x) // 2
                y_cord = (min_white_y + max_white_y) // 2
            else:
                x_cord = circles[1]
                y_cord = circles[0]
        gray_img = cv2.cvtColor(filtered_image_red, cv2.COLOR_BGR2GRAY)
        red_pixels = np.where(gray_img > 0)
        test = np.float32(np.vstack(red_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(test.T, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if len(center) == 1:
            k_means_center = center[0]
            x_cord = int(k_means_center[0])
            y_cord = int(k_means_center[1])

        if not robust:
            return y_cord, x_cord
        else:
            gray_img = cv2.cvtColor(filtered_image_red, cv2.COLOR_BGR2GRAY)
            gray_img[gray_img > 0] = 255
            edges = cv2.Canny(gray_img, 0, 0, apertureSize=3)

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15,
                                    minLineLength=8,
                                    maxLineGap=8)

            if lines is not None:
                lines = lines.reshape(lines.shape[0], -1)
                min_length_lines = [l for l in lines if calc_distance((l[0], l[1]), (l[2], l[3])) > 5]
                acceptable_angles_in_degrees = [0]
                acceptable_angles_in_radians = [(i * (np.pi / 180)) for i in acceptable_angles_in_degrees]
                temp = {(i[0], i[1], i[2], i[3]): np.round(np.abs(calc_angle_in_degrees((i[0], i[1]), (i[2], i[3]))))
                        for i in
                        min_length_lines}
                filtered_lines = []
                for key, val in temp.items():
                    if val in acceptable_angles_in_degrees:
                        filtered_lines.append(key)
            filtered_lines = np.asarray(filtered_lines)
            test = np.float32(np.vstack((filtered_lines[:, [0, 1]], filtered_lines[:, [2, 3]])))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            t, label, centers = cv2.kmeans(test, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            if centers is not None:
                if len(center) > 0:
                    centers = centers[0]

            circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=1,
                                       param2=24,
                                       minRadius=18, maxRadius=0)
            if circles is not None:
                if len(circles) > 0:
                    circles = np.ravel(circles)
                circle_x = circles[0]
                circle_y = circles[1]
                circle_radius = circles[2]

                dst = np.sqrt(
                    ((circle_y - cords_for_all_white_points[0]) ** 2 + (circle_x - cords_for_all_white_points[1]) ** 2))
                in_circle = dst <= circle_radius
                all_x_cords = cords_for_all_white_points[0][in_circle]
                all_y_cords = cords_for_all_white_points[1][in_circle]
                test = np.float32(np.vstack((all_x_cords, all_y_cords)))
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                if len(test) > 1:
                    _, label, center = cv2.kmeans(test.T, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    if center is not None:
                        if len(center) > 0:
                            center = center[0]

                        return center[1], center[0]
                else:
                    return None
            else:
                test = np.float32(np.vstack((cords_for_all_white_points[0], cords_for_all_white_points[1]))).T
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                if len(test) > 1:
                    t, label, centers = cv2.kmeans(test, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    cluster_count = np.unique(label, return_counts=True)
                    centroid_used = centers[np.argmax(cluster_count[0])]
                    return centroid_used[1], centroid_used[0]
                else:
                    return None
    else:
        # figure_out_diameter
        img_in = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        grayscale_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        grayscale_image[grayscale_image > 0] = 255
        all_red_pixels = np.where(grayscale_image == 255)
        test = np.float32(np.vstack((all_red_pixels[0], all_red_pixels[1]))).T
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(test, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        cords = center[0][1], center[0][0]

        return cords


def find_traffic_signal(img_in):
    # Sign primarily red with white letters, octagonal in shape
    gray_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image > 0] = 255
    pixels = np.where(gray_image > 0)
    pt1 = (pixels[1].min(), pixels[0].min())
    pt2 = (pixels[1].max(), pixels[0].min())
    pt3 = (pixels[1].min(), pixels[0].max())
    pt4 = (pixels[1].max(), pixels[0].max())
    calc_center = (pixels[1].min() + pixels[1].max()) // 2, (pixels[0].min() + pixels[0].max()) // 2

    test = np.float32(np.vstack(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(test.T, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    k_means_center = (0, 0)
    if center is not None:
        if len(center) > 0:
            k_means_center = np.round(center[0])
            return k_means_center[1], k_means_center[0]
    return calc_center


def traffic_sign_detection(img_in, is_challenge_problem=False):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    if not is_challenge_problem:
        sign_search = set()
        img_in = cv2.GaussianBlur(img_in, (3, 3), 1)
        img_in = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        # Find sun and remove
        sun_mask = create_yellow_hsv_mask(hsv_image)
        yellow_pixel_count = np.count_nonzero(sun_mask == 255)
        yellow_for_sun_removal = cv2.bitwise_and(img_in, img_in, mask=sun_mask)
        gray_yellow = cv2.cvtColor(yellow_for_sun_removal, cv2.COLOR_BGR2GRAY)
        gray_yellow[gray_yellow > 0] = 255
        circles = cv2.HoughCircles(gray_yellow, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10,
                                   param2=16,
                                   minRadius=20, maxRadius=0)
        temp_img = np.copy(img_in)
        if circles is not None:
            if len(circles) > 0:
                circles = circles[0][:]
            sun = circles[0]
            all_yellow_pixels = np.where(gray_yellow > 0)
            dist = np.sqrt(((sun[1] - all_yellow_pixels[0]) ** 2 + (sun[0] - all_yellow_pixels[1]) ** 2))
            x_in_sun = all_yellow_pixels[0][dist < sun[2]]
            y_in_sun = all_yellow_pixels[1][dist < sun[2]]
            temp_img[x_in_sun, y_in_sun] = [0, 0, 255]
        yellow_mask = create_yellow_hsv_mask(cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV))
        yellow_pixel_count = np.count_nonzero(yellow_mask == 255)
        image_filtered_for_warning_sign = cv2.bitwise_and(img_in, img_in, mask=yellow_mask)
        warning_cords = warning_sign_detection(image_filtered_for_warning_sign, robust=True)
        if warning_cords is None:
            if "Warning Sign" in sign_search:
                sign_search.remove("Warning Sign")
        else:
            all_pixels_from_yellow_mask = np.where(yellow_mask > 0)
            dist = np.sqrt(((warning_cords[1] - all_pixels_from_yellow_mask[0]) ** 2 + (
                    warning_cords[0] - all_pixels_from_yellow_mask[1]) ** 2))
            a = np.std(dist)
            x_in_sign = all_pixels_from_yellow_mask[0][dist < a]
            y_in_sign = all_pixels_from_yellow_mask[1][dist < a]
        sun_and_warn_mask = sun_mask + yellow_mask
        orange_mask = create_orange_hsv_mask(hsv_image)
        orange_pixel_count = np.count_nonzero(orange_mask == 255)
        black_mask = create_black_hsv_mask(hsv_image)
        black_pixel_count = np.count_nonzero(black_mask == 255)
        sun_warn_mask = sun_and_warn_mask + orange_mask
        image_filtered_for_construction_sign = cv2.bitwise_and(img_in, img_in, mask=orange_mask)
        temp_img = np.copy(img_in)
        temp_img = cv2.bitwise_and(temp_img, temp_img, mask=sun_warn_mask)
        green_mask = create_green_hsv_mask(hsv_image)
        green_pixel_count = np.count_nonzero(green_mask == 255)
        white_mask = create_white_hsv_mask(hsv_image)
        white_pixel_count = np.count_nonzero(white_mask == 255)
        image_filtered_for_traffic_light = cv2.bitwise_and(img_in, img_in, mask=black_mask)
        traffic_black_pixel = image_filtered_for_traffic_light > 0
        traffic_count = np.count_nonzero(traffic_black_pixel)
        if traffic_count == 0:
            traffic_light_cords = None
        else:
            traffic_light_cords = find_traffic_signal(image_filtered_for_traffic_light)
        red_mask_low, red_mask_high = create_red_hsv_mask(hsv_image)
        red_mask = red_mask_low + red_mask_high
        red_pixel_count = np.count_nonzero(red_mask == 255)
        image_filtered_for_no_entry_sign = cv2.bitwise_and(img_in, img_in, mask=white_mask + red_mask)
        image_filtered_for_stop_sign = cv2.bitwise_and(img_in, img_in, mask=white_mask + red_mask)
        stop_cords = stop_sign_detection(image_filtered_for_stop_sign, robust=True)
        image_filtered_for_yield_sign = cv2.bitwise_and(img_in, img_in, mask=red_mask + white_mask)

        no_entry_cords = do_not_enter_sign_detection(image_filtered_for_no_entry_sign, robust=True)
        # if do_not_enter_cords is not None:

        yield_cords = yield_sign_detection(image_filtered_for_yield_sign, robust=True)
        if red_pixel_count > 20:
            if green_pixel_count > 20:
                if "Traffic Light" not in sign_search:
                    sign_search.add("Traffic Light")
            if white_pixel_count > 20:
                if "Stop Sign" not in sign_search:
                    sign_search.add("Stop Sign")
                if "Yield Sign" not in sign_search:
                    sign_search.add("Yield Sign")
                if "Warning Sign" not in sign_search:
                    sign_search.add("Warning Sign")
        if orange_pixel_count > 20:
            if "Construction Sign" not in sign_search:
                sign_search.add("Construction Sign")
        if yellow_pixel_count > 20:
            if "Warning Sign" not in sign_search:
                sign_search.add("Warning Sign")
        if yield_cords is None:
            if "Yield Sign" in sign_search:
                sign_search.remove("Yield Sign")
        construction_cords = construction_sign_detection(image_filtered_for_construction_sign, robust=True)
        if construction_cords is None:
            if "Construction Sign" in sign_search:
                sign_search.remove("Construction Sign")
        temp = np.copy(img_in)
        if construction_cords is not None:
            cv2.circle(temp, center=(construction_cords[0], construction_cords[1]), color=[255, 0, 0], radius=3,
                       thickness=3)
        if warning_cords is not None:
            cv2.circle(temp, center=(warning_cords[0], warning_cords[1]), color=[255, 0, 0], radius=3, thickness=3)
        if yield_cords is not None:
            cv2.circle(temp, center=(yield_cords[0], yield_cords[1]), color=[255, 0, 0], radius=3, thickness=3)

        if no_entry_cords is not None:
            cv2.circle(temp, center=(no_entry_cords[0], no_entry_cords[1]), color=[255, 0, 0], radius=3, thickness=3)
        if traffic_light_cords is not None:
            cv2.circle(temp, center=(traffic_light_cords[0], traffic_light_cords[1]), color=[255, 0, 0], radius=3,
                       thickness=3)
        if stop_cords is not None:
            cv2.circle(temp, center=(stop_cords[0], stop_cords[1]), color=[255, 0, 0], radius=3, thickness=3)

        results = {}
        if warning_cords is not None:
            results['warning'] = warning_cords
        if traffic_light_cords is not None:
            results['traffic light'] = traffic_light_cords
        if no_entry_cords is not None:
            results['no_entry'] = no_entry_cords
        if stop_cords is not None:
            results['stop'] = stop_cords
        if yield_cords is not None:
            results['yield'] = yield_cords
        if construction_cords is not None:
            results['construction'] = construction_cords
        return results
    else:
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        temp_img = np.copy(img_in)
        yellow_mask = create_yellow_hsv_mask(cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV))
        yellow_pixel_count = np.count_nonzero(yellow_mask == 255)
        image_filtered_for_warning_sign = cv2.bitwise_and(img_in, img_in, mask=yellow_mask)
        warning_cords = warning_sign_detection(image_filtered_for_warning_sign, robust=True,
                                               is_challenge=is_challenge_problem)


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    return traffic_sign_detection(img_in=img_in)


def traffic_sign_detection_challenge(img_in, problem="a1"):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    unmodified = np.copy(img_in)
    if problem == "a1":
        img_in = cv2.GaussianBlur(img_in, (3, 3), 1)
        img_in = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        yellow_mask = create_yellow_hsv_mask(hsv_image, is_challenge=True)
        yellow_mask_kangaroo = cv2.inRange(hsv_image, np.array([0, 180, 200]), np.array([39, 255, 234]))

        red_mask_dne = cv2.inRange(hsv_image, np.array([22, 255, 0]), np.array([255, 255, 255])) + \
                       cv2.inRange(hsv_image, np.array([172, 146, 142]), np.array([177, 255, 213]))
        red_filtered_image = cv2.bitwise_and(img_in, img_in, mask=red_mask_dne)
        dne_cords = do_not_enter_sign_detection(red_filtered_image, robust=True, is_challenge=True)
        temp_image = np.copy(unmodified)
        center = (int(dne_cords[0]), int(dne_cords[1]))

        # Using provided helper to add labels to image, Modified to be larger due to larger images.
        cv2.drawMarker(temp_image, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=200, thickness=75)
        text = "{}: ({}, {})".format("no_entry", center[0], center[1])
        place_text(text, center, temp_image)
        cv2.imwrite("output/{}.png".format("ps2-5-a-1"), temp_image)

        return {"no_entry": dne_cords}
    elif problem == "a2" or problem == "a3":
        img_in = cv2.GaussianBlur(img_in, (3, 3), 1)
        img_in = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        hsv_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
        yellow_mask = create_yellow_hsv_mask(hsv_image, is_challenge=True)
        yellow_mask_kangaroo = cv2.inRange(hsv_image, np.array([0, 180, 172]), np.array([39, 255, 234]))
        yellow_filtered_image = cv2.bitwise_and(img_in, img_in, mask=yellow_mask_kangaroo)
        grayscale_image = cv2.cvtColor(yellow_filtered_image, cv2.COLOR_BGR2GRAY)
        grayscale_image[grayscale_image > 0] = 255
        all_yellow_pixels = np.where(grayscale_image == 255)
        pixel_array = np.vstack((all_yellow_pixels[0], all_yellow_pixels[1])).T
        min_y = pixel_array[:, 1].min()
        max_y = pixel_array[:, 1].max()
        x_cords_for_min_y = np.mean(pixel_array[:, 0][pixel_array[:, 1] == min_y])
        x_cords_for_max_y = np.mean(pixel_array[:, 0][pixel_array[:, 1] == max_y])
        mid_x = int((x_cords_for_min_y + x_cords_for_max_y) // 2)
        mid_y = int(min_y + (max_y - min_y) // 2)
        test = np.float32(pixel_array)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, label, center_a = cv2.kmeans(test, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = (int(center_a[0][1]), int(center_a[0][0]))

        temp_image = np.copy(unmodified)
        if problem == 'a2':
            # Using provided helper to add labels to image, Modified to be larger due to larger images.
            cv2.drawMarker(temp_image, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=200, thickness=50)
            text = "{}: ({}, {})".format("warning", center[0], center[1])
            modified_center_to_move_text_placement = center[0] - 75, center[1] - 50
            place_text(text, modified_center_to_move_text_placement, temp_image)
            cv2.imwrite("output/{}.png".format("ps2-5-a-2"), temp_image)
            return {"warning": center}
        elif problem == "a3":
            # Using provided helper to add labels to image, Modified to be larger due to larger images.
            cv2.drawMarker(temp_image, center, marker_color, markerType=cv2.MARKER_CROSS, markerSize=100, thickness=20)
            text = "{}: ({}, {})".format("warning", center[0], center[1])
            modified_center_to_move_text_placement = center[0] + 600, center[1] - 100
            place_text(text, modified_center_to_move_text_placement, temp_image)
            cv2.imwrite("output/{}.png".format("ps2-5-a-3"), temp_image)
            return {"warning": center}
        else:
            return
        return
    elif problem == "a3":
        return
    else:
        return


################################ CHANGE BELOW FOR MORE CUSTOMIZATION #######################
""" The functions below are used for each individual part of the report section.

Feel free to change the return statements but ensure that the return type remains the same 
for the autograder. 

"""


# Part 2 outputs
def ps2_2_a_1(img_in):
    return do_not_enter_sign_detection(img_in)


def ps2_2_a_2(img_in):
    return stop_sign_detection(img_in)


def ps2_2_a_3(img_in):
    return construction_sign_detection(img_in)


def ps2_2_a_4(img_in):
    return warning_sign_detection(img_in)


def ps2_2_a_5(img_in):
    return yield_sign_detection(img_in)


# Part 3 outputs
def ps2_3_a_1(img_in):
    return traffic_sign_detection(img_in)


def ps2_3_a_2(img_in):
    return traffic_sign_detection(img_in)


# Part 4 outputs
def ps2_4_a_1(img_in):
    return traffic_sign_detection_noisy(img_in)


def ps2_4_a_2(img_in):
    return traffic_sign_detection_noisy(img_in)


# Part 5 outputs
def ps2_5_a(img_in):
    return traffic_sign_detection_challenge(img_in)


def ps2_5_a_1(img_in):
    return traffic_sign_detection_challenge(img_in, problem="a1")


def ps2_5_a_2(img_in):
    # Added by me for extra testing
    return traffic_sign_detection_challenge(img_in, problem='a2')


def ps2_5_a_3(img_in):
    # Added by me for extra testing
    return traffic_sign_detection_challenge(img_in, problem='a3')
