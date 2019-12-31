import numpy as np
import sys
import imageio
import os
import cv2
import skimage
from pathlib import Path
from scipy import signal
from matplotlib import pyplot as plt

from glob import glob

'''
https://sites.google.com/site/obstructionfreephotography/
https://ai.googleblog.com/2017/04/photoscan-taking-glare-free-pictures-of.html
https://www.youtube.com/watch?v=KoMTYnlNNnc
https://youtu.be/3JwoW7fQXWM
'''


# noinspection PyMethodMayBeStatic
class ImageObstructionClean:
    np.random.seed(5)

    def __init__(self, images=None, image_paths=None):
        self.images = images
        self.blurred_images = None
        self.image_paths = image_paths
        self.edges = None
        self.normalizedEdges = None
        self.FeatureParams = dict(maxCorners=5, qualityLevel=0.01, minDistance=7, blockSize=5)
        self.Lk_params = dict(winSize=(3, 3), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.0000001))
        self.color = np.random.randint(0, 255, (self.FeatureParams["maxCorners"], 3))
        self.Transformation_Matrix = []
        self.LoadImages()
        # self.Routine()
        self.TEST()

    @staticmethod
    def getCurrentWorkingDirectory():
        try:
            return os.getcwd()
        except Exception as GetCurrentWorkingDirectoryException:
            print("Exception occurred while attempting to execute function getCurrentWorkingDirectory. \n",
                  GetCurrentWorkingDirectoryException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def LoadImages(self):
        try:
            if (self.images is None) and (self.image_paths is not None):
                # LoadImages
                self.images = [cv2.imread(img, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for img in self.image_paths]
            if self.edges is None:
                self.FindEdges()
        except Exception as LoadImagesException:
            print("Exception occurred while attempting to load images. \n", LoadImagesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def FindEdges(self):
        try:
            if self.edges is not None:
                return
            color = (0, 0, 255)
            lineThickness = 1
            fontSize = 0.75
            if self.blurred_images is None:
                self.blurred_images = [cv2.GaussianBlur(img, (9, 9), 0) for img in self.images]
            self.edges = [cv2.Canny(img, 22, 38) for img in self.images]
            for i in range(len(self.edges)):
                cv2.imwrite((self.getCurrentWorkingDirectory() + "/Edges") + "/frame{0:04d}.png".format(i),
                            self.edges[i])
            return
        except Exception as FindEdgesException:
            print("Exception occurred while attempting to find edges. \n", FindEdgesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def FindCorners(self, blockSize, apertureSize, k, border=None):
        try:
            test = cv2.cornerHarris(self.edges[0], blockSize=blockSize, ksize=apertureSize, k=k)
            test2 = cv2.goodFeaturesToTrack(self.edges[0], mask=None, **self.FeatureParams)
            result = self.edges[0].copy()
            result = (result / result.max()) * 255.0
            result = result.astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            for corner in test2:
                cv2.circle(result, center=(corner[0][0], corner[0][1]), radius=10, color=(0, 0, 255), thickness=5)
            if test.dtype != np.uint8:
                test = test.astype(np.uint8)
            return

        except Exception as err:
            print("Exception occurred. \n", err)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def Routine(self):
        try:

            # Initial Motion Estimation
            # region Find Edges
            self.FindEdges()
            # endregion
            t = self.edges[0].ravel()
            result = cv2.findHomography(self.edges[2], self.edges[0], method=cv2.RANSAC, ransacReprojThreshold=5.0)

            print()
            # region Find Corners on the Edges
            # self.FindCorners(blockSize=2, apertureSize=3, k=0.04)
            # endregion

            # region Markov Random Field (MRF)

            # endregion

            return
        except Exception as RoutineException:
            print("Exception occurred while attempting to execute the Routine. \n", RoutineException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def reflectionRoutine(self):
        return

    def arrowStuff(self, imgs):
        try:
            ksize = (9, 9)  # size of the window w(u, v)
            sigma = 1.  # standard deviation of a Gaussian filter w(u, v)
            kappa = 0.01  # Harris-Stephens corner score parameter
            threshold_ratio = 0.1  # "corners" are larger than threshold_ratio * max(Mc)
            for i in range(len(imgs)):
                if len(imgs[i].shape) == 3:
                    gray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
                else:
                    gray = imgs[i]
                Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                M_Ixx = cv2.GaussianBlur(Ix * Ix, ksize, sigma)
                M_Iyy = cv2.GaussianBlur(Iy * Iy, ksize, sigma)
                M_Ixy = cv2.GaussianBlur(Ix * Iy, ksize, sigma)

                R = (M_Ixx * M_Iyy - M_Ixy * M_Ixy) - kappa * (M_Ixx + M_Iyy) ** 2
                det = (M_Ixx * M_Iyy - M_Ixy * M_Ixy)

                y, x = np.where((R == cv2.dilate(R, np.ones(ksize))) & (R > threshold_ratio * R.max()))
                plt.figure(figsize=(16, 12))
                # plt.scatter(x, y, s=1, c='r');  # plot the keypoints
                plt.quiver(x, y, Ix[y, x], Iy[y, x], color='r',
                           width=0.001)  # plot the gradient magnitude & direction
                plt.axis("off")
                plt.imshow(gray, cmap="gray")
                plt.savefig("ArrowTest/frame{0:04d}.png".format(i))

        except Exception as ArrowStuffException:
            print("Exception occurred while attempting to execute ArrowStuff. \n", ArrowStuffException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def OpticalFlow(self, imgs):
        try:
            # Source https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
            # feature_params = dict(maxCorners=500,
            # 					   qualityLevel=0.3,
            # 					   minDistance=7,
            # 					   blockSize=15)

            # Parameters for lucas kanade optical flow
            # lk_params = dict(winSize=(12, 12),
            # 				  maxLevel=4,
            # 				  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Create some random colors
            np.random.seed(5)
            # color = np.random.randint(0, 255, (feature_params["maxCorners"], 3))
            old_frame = imgs[0]
            if len(old_frame.shape) == 3:
                old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            else:
                old_gray = old_frame
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.FeatureParams)
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            for img in range(1, len(imgs)):
                # self.arrowStuff()
                frame = imgs[img]
                if len(frame.shape) == 3:
                    frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.Lk_params)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
                new_img = cv2.add(frame, mask)

                cv2.imshow('frame', new_img)
                cv2.imwrite("OpticalFlow/frame{0:04d}.png".format(img), new_img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            cv2.destroyAllWindows()
            return
        except Exception as OpticalFlowShiTomasiException:
            print("Exception occurred while attempting to execute OpticalFlowShiTomasi. \n",
                  OpticalFlowShiTomasiException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def OpticalFlowDense(self, imgs):
        try:
            # Source https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
            # Parameters for lucas kanade optical flow
            # lk_params = dict(winSize=(25, 25),
            # 				  maxLevel=6,
            # 				  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            frame1 = imgs[0]
            if len(frame1.shape) == 3:
                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            else:
                prvs = frame1
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            for img in range(1, len(imgs)):
                frame2 = imgs[img]
                if len(frame2.shape) == 3:
                    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                else:
                    next = frame2
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 9, 3, 5, 1.2, 0)
                cv2.imshow("Flow", flow.astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                bgr = hsv
                cv2.imshow('frame2', bgr)
                cv2.imwrite("DenseOpticalFlow/frame{0:04d}.png".format(img), bgr)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png', frame2)
                    cv2.imwrite('opticalhsv.png', bgr)
                prvs = next
            cv2.destroyAllWindows()
            return
        except Exception as OpticalFlowLucasKanadeException:
            print("Exception occurred while attempting to execute OpticalFlowLucasKanade. \n",
                  OpticalFlowLucasKanadeException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def getEdgePixels(self, input_edges=None, row_index=0):
        try:
            if input_edges is not None:
                cords = np.where(input_edges > 0)
                t, p = np.where(input_edges > 0)
                print()
                rows = cords[0]
                columns = cords[1]
                if row_index == 0:
                    combined_rows_and_columns = np.column_stack((rows, columns))
                else:
                    combined_rows_and_columns = np.column_stack((columns, rows))
                return rows, columns, combined_rows_and_columns
            else:
                return
        except Exception as GetEdgePixelsException:
            print("Exception occurred while attempting to execute GetEdgePixels. \n",
                  GetEdgePixelsException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def TestRoutine(self):
        try:
            print()
            # Find Edges
            self.FindEdges()
            
            starting_image = self.images[2]
            ending_image = self.images[1]
            
            starting_edges = self.edges[2]
            ending_edges = self.edges[1]
            mask = np.zeros_like(starting_image)
            starting_edge_rows, starting_edges_cols, starting_edges_pixels_ColRow = self.getEdgePixels(starting_edges,
                                                                                                       row_index=1)

            nextPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImg=starting_image, nextImg=ending_image,
                                                                 nextPts=None,
                                                                 prevPts=starting_edges_pixels_ColRow.astype(np.float32),
                                                                 **self.Lk_params)
            
            good_old_pixels = starting_edges_pixels_ColRow[np.where(status.astype(bool)), :][0]
            good_new_pixels = nextPoints[np.where(status.astype(bool)), :][0]
            
            msk = status.astype(bool)
            start_matched_pixels = starting_edges_pixels_ColRow[np.where(status.astype(bool)), :][0]
            end_matched_pixels = nextPoints[np.where(status.astype(bool)), :][0]
            print()

            background_transformation_matrix, background_homography_mask = cv2.findHomography(srcPoints=start_matched_pixels,
                                                                    dstPoints=end_matched_pixels,
                                                                    method=cv2.RANSAC, ransacReprojThreshold=1.0)
            background_homography_mask = background_homography_mask.astype(np.bool)
            background_pixels = start_matched_pixels[np.where(background_homography_mask), :]
            start_matched_pixels_filtered = start_matched_pixels[np.where(~background_homography_mask), :][0]
            end_matched_pixels_filtered = end_matched_pixels[np.where(~background_homography_mask), :][0]
            obstruction_transformation_matrix, obstruction_homography_mask = cv2.findHomography(
                srcPoints=start_matched_pixels_filtered,
                dstPoints=end_matched_pixels_filtered,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0)
            
            temp_result = np.zeros(shape=self.edges[0].shape)
            back_cols = background_pixels[0][:, 0]
            back_rows = background_pixels[0][:, 1]
            temp_result[[back_rows, back_cols]] = 255
            cv2.imwrite("Background.png", temp_result)
            print()
            result1 = cv2.warpPerspective(starting_image,
                                          background_transformation_matrix,
                                          (starting_image.shape[1], starting_image.shape[0]))
            result2 = cv2.warpPerspective(starting_image,
                                          obstruction_transformation_matrix,
                                          (starting_image.shape[1], starting_image.shape[0]))
            cv2.imwrite("Results_1.png", result1)
            cv2.imwrite("Results_2.png", result2)

            result_background = np.zeros_like(result1).astype(np.float64)
            result_obstruction = np.zeros_like(result1).astype(np.float64)
            array_of_background = []
            array_of_background_split_channel = [[], [], []]
            array_of_obstruction = []
            array_of_obstruction_split_channel = [[], [], []]

            for i in range(len(self.images)):
                temp_background = cv2.warpPerspective(self.images[i], background_transformation_matrix,
                                                      (self.images[i].shape[1], self.images[i].shape[0]))
                temp_obstruction = cv2.warpPerspective(self.images[i], background_transformation_matrix,
                                                       (self.images[i].shape[1], self.images[i].shape[0]))
                array_of_background.append(temp_background)
                back_b, back_g, back_r = cv2.split(temp_background)
                obstruct_b, obstruct_g, obstruct_r = cv2.split(temp_obstruction)
                array_of_background_split_channel[0].append(back_b)
                array_of_background_split_channel[1].append(back_g)
                array_of_background_split_channel[2].append(back_r)
                array_of_obstruction_split_channel[0].append(obstruct_b)
                array_of_obstruction_split_channel[1].append(obstruct_g)
                array_of_obstruction_split_channel[2].append(obstruct_r)
                array_of_obstruction.append(temp_obstruction)
                result_background += (self.images[2] * 0.2 - temp_background * 0.2)
                result_obstruction -= (self.images[2] * 0.2 - temp_obstruction * 0.2)
                cv2.imwrite("BackGround_{}.png".format(i), result_background)
                cv2.imwrite("Obstruction_{}.png".format(i), result_obstruction)
            cv2.imwrite("Final_BackGround_{}.png".format(i), result_background)
            cv2.imwrite("Final_Obstruction_{}.png".format(i), result_obstruction)
            test = self.images[0] + result_obstruction
            cv2.imwrite("Final_Result.png", test)
            shape_of_array = array_of_background[0].shape
            fill_array = np.zeros_like(array_of_background[0])
            min_b_channel = np.asarray(array_of_background_split_channel[0]).min(axis=0)
            min_g_channel = np.asarray(array_of_background_split_channel[1]).min(axis=0)
            min_r_channel = np.asarray(array_of_background_split_channel[2]).min(axis=0)
            Min_Pixel_Image = np.asarray(array_of_background).min(axis=0)
            Min_Intensity_Image = cv2.merge((min_b_channel, min_g_channel, min_r_channel))
            cv2.imwrite("Min_Pixel_Image.jpg", Min_Pixel_Image)
            cv2.imwrite("Min_Intensity_Image.jpg", Min_Intensity_Image)
            return

        except Exception as TestRoutineException:
            print("Exception occurred while attempting to execute TestRoutine. \n", TestRoutineException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def findHomography(self, image_1_kp, image_2_kp, matches):
        """Returns the homography describing the transformation between the
        keypoints of image 1 and image 2.

            ************************************************************
              Before you start this function, read the documentation
                      for cv2.DMatch, and cv2.findHomography
            ************************************************************

        Follow these steps:

            1. Iterate through matches and store the coordinates for each
               matching keypoint in the corresponding array (e.g., the
               location of keypoints from image_1_kp should be stored in
               image_1_points).

                NOTE: Image 1 is your "query" image, and image 2 is your
                      "train" image. Therefore, you index into image_1_kp
                      using `match.queryIdx`, and index into image_2_kp
                      using `match.trainIdx`.

            2. Call cv2.findHomography() and pass in image_1_points and
               image_2_points, using method=cv2.RANSAC and
               ransacReprojThreshold=5.0.

            3. cv2.findHomography() returns two values: the homography and
               a mask. Ignore the mask and return the homography.

        Parameters
        ----------
        image_1_kp : list<cv2.KeyPoint>
            A list of keypoint descriptors in the first image

        image_2_kp : list<cv2.KeyPoint>
            A list of keypoint descriptors in the second image

        matches : list<cv2.DMatch>
            A list of matches between the keypoint descriptor lists

        Returns
        -------
        numpy.ndarray(dtype=np.float64)
            A 3x3 array defining a homography transform between image_1 and image_2
        """
        try:
            image_1_points = []
            image_2_points = []
            for match in matches:
                image_1_points.append(image_1_kp[match.queryIdx].pt)
                image_2_points.append(image_2_kp[match.trainIdx].pt)
            image_1_points = np.asarray(image_1_points)
            image_2_points = np.asarray(image_2_points)

            homography_matrix, homography_mask = cv2.findHomography(image_1_points, image_2_points,
                                                                    method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if homography_matrix.dtype != np.float64:
                homography_matrix = np.float64(homography_matrix)

            return homography_matrix, homography_mask

        except Exception as FindHomographyException:
            print("Exception while processing 'findHomography'. \n", FindHomographyException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def findMatchesBetweenImages(self, image_1, image_2, num_matches):
        """Return the top list of matches between two input images.

        Parameters
        ----------
        image_1 : numpy.ndarray
            The first image (can be a grayscale or color image)

        image_2 : numpy.ndarray
            The second image (can be a grayscale or color image)

        num_matches : int
            The number of keypoint matches to find. If there are not enough,
            return as many matches as you can.

        Returns
        -------
        image_1_kp : list<cv2.KeyPoint>
            A list of keypoint descriptors from image_1

        image_2_kp : list<cv2.KeyPoint>
            A list of keypoint descriptors from image_2

        matches : list<cv2.DMatch>
            A list of the top num_matches matches between the keypoint descriptor
            lists from image_1 and image_2

        Notes
        -----
            (1) You will not be graded for this function.
        """
        try:
            if type(image_1) != np.ndarray:
                image_1 = np.asarray(image_1)
            if type(image_2) != np.ndarray:
                image_2 = np.asarray(image_2)
            num_matches = int(num_matches)

            feat_detector = cv2.ORB_create(nfeatures=500)
            if image_1.dtype != np.uint8:
                image_1 = np.uint8(image_1)
            image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
            image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
            bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bfm.match(image_1_desc, image_2_desc),
                             key=lambda x: x.distance)[:num_matches]
            return image_1_kp, image_2_kp, matches
        except Exception as FindMatchesBetweenImagesException:
            print("Exception while processing 'findMatchesBetweenImages'. \n", FindMatchesBetweenImagesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def GetGradients(self):
        try:
            temp = cv2.GaussianBlur(self.images[0], (5, 5), 0)
            dx = cv2.Sobel(temp, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(temp, cv2.CV_64F, 0, 1, ksize=3)

            test_dx = np.absolute(dx)
            test_dy = np.absolute(dy)
            norm_test_dx = (test_dx / test_dx.max()) * 255
            norm_test_dy = (test_dy / test_dy.max()) * 255
            cv2.imwrite("Normalized_Gradients_Dx.jpg", norm_test_dx)
            cv2.imwrite("Normalized_Gradients_Dy.jpg", norm_test_dy)
            cv2.imshow("Normalized Dx", norm_test_dx.astype(np.uint8))
            cv2.imshow("Normalized Dy", norm_test_dy.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        except Exception as GetGradientsException:
            print("Exception occurred while attempting to execute GetGradients. \n", GetGradientsException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def TEST(self):
        # self.FindCorners(blockSize=2, apertureSize=3, k=0.04)
        self.TestRoutine()

        # self.arrowStuff(self.edges)
        # self.GetGradients()
        print()


def getCurrentWorkingDirectory():
    try:
        return os.getcwd()
    except Exception as GetCurrentWorkingDirectoryException:
        print("Exception occurred while attempting to execute function getCurrentWorkingDirectory. \n",
              GetCurrentWorkingDirectoryException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def getFramesFromVideo(pathToVideo, saveDirectory):
    try:
        # Source https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
        # I did not make this, I used it to get the frames from the video used in the paper
        # cv2.readOpticalFlow()
        video_capture = cv2.VideoCapture(pathToVideo)
        success, image = video_capture.read()
        count = 0
        while success:
            # save frame as JPEG file
            # 'frame{0:04d}.png'.format(idx)
            cv2.imwrite(saveDirectory + "/frame{0:04d}.png".format(count), image)
            success, image = video_capture.read()
            count += 1
        return
    except Exception as GetFramesFromVideoException:
        print("Exception occurred while executing GetFramesFromVideo. \n", GetFramesFromVideoException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def optical_flow(I1g, I2g, window_size, tau=1e-2):
    # https: // sandipanweb.wordpress.com / 2018 / 02 / 25 / implementing - lucas - kanade - optical - flow - algorithm - in -python /
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = window_size/2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            #b = ... # get b here
            #A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ... # get velocity here
            u[i,j]=nu[0]
            v[i,j]=nu[1]

    return (u,v)


def getImageFileNames(image_dir):
    try:
        extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                      'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

        search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
        image_files = sorted(sum(map(glob, search_paths), []))
        return image_files
    except Exception as GetImageFileNamesException:
        print("Exception occurred while attempting to get the image file names. \n", GetImageFileNamesException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def generateGif(files, dur=0.1, name=None, ATWORK=False):
    try:
        dir = "C:/Users/joshu/"
        if ATWORK:
            dir = "C:/Users/Josh.Adams/"
        with imageio.get_writer("{}OneDrive - Georgia Institute of "
                                "Technology/Georgia-Tech/CS 6475 - Computational "
                                "Photography/assignments/Final_Project/{}.gif".format(dir, name),
                                mode="I", duration=dur) as writer:
            for filename in files:
                image = imageio.imread(filename)
                writer.append_data(image)
        return
    except Exception as GenerateGifException:
        print("Exception occurred while executing GenerateGif. \n", GenerateGifException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def resizeImages(filePaths, percent):
    try:
        img_dir = Path(filePaths[0]).parents[0]
        output_directory_percent = os.path.join(str(img_dir), "Resized_{}%_Images".format(percent * 100))
        # Check if directories exist and create if they do not
        # Source https://stackoverflow.com/questions/31008598/python-check-if-a-directory-exists-then-create-it-if-necessary-and-save-graph-t?noredirect=1&lq=1

        if not os.path.exists(output_directory_percent):
            os.mkdir(output_directory_percent)
        for img in filePaths:
            current_file_name = img.split("\\")[-1]
            temp_out_1 = os.path.join(output_directory_percent, current_file_name)
            # Read Image
            temp_img = cv2.imread(img, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
            # Resize Images
            resized_img_1 = cv2.resize(temp_img, None, fx=percent, fy=percent)
            # Save Images
            cv2.imwrite(temp_out_1, resized_img_1)
        return output_directory_percent
    except Exception as ResizeImageException:
        print("Exception occurred while attempting to resize images. \n", ResizeImageException)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
