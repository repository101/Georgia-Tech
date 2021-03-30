import cv2
import ps5
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0
    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # video = cv2.VideoWriter('./noise.avi', fourcc, float(24), (480, 360))
    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        # out_frame = frame.copy()
        # pf.render(out_frame)
        # video.write(out_frame)

        # if True:  # For debugging, it displays every frame
        #     visualize_filter(pf=pf, frame=frame)
        #     out_frame = frame.copy()
        #     pf.render(out_frame)
        #     visualize_filter(pf=pf, frame=out_frame)
        #     cv2.imshow('Tracking', out_frame)
        #     cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            frame_out = pf.render(frame_out, is_part_4=True)
            cv2.imshow("t", frame_out)
            cv2.waitKey(1)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    # video.release()
    cv2.destroyAllWindows()
    return 0


def run_kalman_filter(filter_class,
                      imgs_dir,
                      noise,
                      sensor,
                      save_frames={},
                      template_loc=None,
                      Q=0.1 * np.eye(4),
                      R=0.1 * np.eye(2)):
    kf = filter_class(template_loc['x'], template_loc['y'], Q, R)

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0

def run_kalman_filter_pt_5(filter_class,
                      imgs_dir,
                      noise,
                      sensor,
                      template_loc=None,
                      Q=0.1 * np.eye(4),
                      R=0.1 * np.eye(2)):
    kf = filter_class(template_loc['x'], template_loc['y'], Q, R)

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0

def visualize_filter(pf, frame):
    out_frame = frame.copy()
    pf.render(out_frame)
    # if pf.previous_n_templates_averaged is not None:
    #     color_template = np.copy(pf.previous_n_templates_averaged).astype(np.uint8)
    # else:
    color_template = np.copy(pf.template)
    color_template = cv2.cvtColor(color_template, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow('template', flags=cv2.WINDOW_NORMAL)
    new_size_width = out_frame.shape[1] + pf.template_width+1
    new_size_height = out_frame.shape[0]
    new_img = np.zeros(shape=(new_size_height+1, new_size_width+1, 3), dtype=np.uint8)
    new_img[0:out_frame.shape[0], 0:out_frame.shape[1], :] = out_frame
    new_img[0:color_template.shape[0], out_frame.shape[1]:out_frame.shape[1]+color_template.shape[1], :] = color_template
    # cv2.imshow('test', out_frame)
    cv2.imshow('template', new_img)
    cv2.resizeWindow("template", width=new_size_width, height=new_size_height)
    # cv2.moveWindow("template", 960, 540)
    cv2.waitKey(1)
    
    
def visualize_filter_pt_5(pf, frame, pedestrian="1", show_weights=False, return_frame=False):
    out_frame = frame.copy()
    pf.render(out_frame)
    # if pf.previous_n_templates_averaged is not None:
    #     color_template = np.copy(pf.previous_n_templates_averaged).astype(np.uint8)
    # else:
    color_template = np.copy(pf.template)
    if len(color_template.shape) <= 2:
        color_template = cv2.cvtColor(color_template.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if not return_frame:
        cv2.namedWindow(f'template_{pedestrian}', flags=cv2.WINDOW_NORMAL)
    new_size_width = out_frame.shape[1] + pf.template_width
    new_size_height = out_frame.shape[0]
    if not show_weights:
        new_img = np.zeros(shape=(new_size_height, new_size_width, 3), dtype=np.uint8)
        new_img[0:out_frame.shape[0], 0:out_frame.shape[1], :] = out_frame
        new_img[0:color_template.shape[0], out_frame.shape[1]:out_frame.shape[1]+color_template.shape[1], :] = color_template
    else:
        weights_img = np.zeros(shape=frame.shape[:2])
        new_size_width = out_frame.shape[1] + pf.template_width + weights_img.shape[1]
        for i in range(len(pf.particles)):
            if pf.particles[i][0] < 0 or pf.particles[i][0] > weights_img.shape[1] or pf.particles[i][1] < 0 or pf.particles[i][1] > weights_img.shape[0]:
                continue
            else:
                weights_img[int(pf.particles[i][1]), int(pf.particles[i][0])] = pf.weights[i]
        # k_size = 7
        # weights_img = cv2.filter2D(weights_img, -1, kernel=(np.ones((k_size, k_size)) / k_size ** 2))
        weights_img = (((weights_img - weights_img.min()) / (weights_img.max() - weights_img.min())) * 255).astype(np.uint8)
        weights_img = weights_img.astype(np.uint8)
        new_size_height = out_frame.shape[0]
        new_img = np.zeros(shape=(new_size_height, new_size_width, 3), dtype=np.uint8)
        new_img[0:out_frame.shape[0], 0:out_frame.shape[1], :] = out_frame
        new_img[0:color_template.shape[0], out_frame.shape[1]:out_frame.shape[1]+color_template.shape[1], :] = color_template

        new_img[0:weights_img.shape[0], out_frame.shape[1] + color_template.shape[1]:out_frame.shape[1] + color_template.shape[1] + weights_img.shape[1], :] = cv2.addWeighted(cv2.cvtColor(weights_img, cv2.COLOR_GRAY2BGR),0.75, pf.unedited_color_frame, 0.25, 0)
    if return_frame:
        image_with_watermark = np.copy(new_img)
        cv2.putText(image_with_watermark, "JADAMS334", (200, 425), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=4, color=(255, 255, 255), thickness=5)
        pct = 0.90
        new_img = cv2.addWeighted(new_img, pct, image_with_watermark, 1-pct, 0)
        cv2.imshow("img", new_img)
        cv2.waitKey(1)
        return new_img
    else:
        cv2.imshow(f'template_{pedestrian}', new_img)
        cv2.resizeWindow(f'template_{pedestrian}', width=new_size_width, height=new_size_height)
        cv2.waitKey(1)
