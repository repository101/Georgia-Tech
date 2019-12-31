import numpy as np
import sys
import imageio
import os
import cv2
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
#TODO Look at 12x12 window, maxLvl=5, ePS=1 count = 1e-05 or count = 0.00001
# 12x12 max=7 eps=1 count = 0.01
# 15x15 maxLvL=5, EPS=1, count = 1e-07
# 25x25 maxlvl=1, eps=3 count =0.0001

TESTING = True     # Global boolean used for testing
PARAMTESTING = False    # Value used when testing parameters in finding Edges
USEBLURRED = False      # Value used to set whether we want to blur the images prior to finding edges ( canny edge detection already implements gaussian blur, so images will essentially be blurred twice )
SWAP_PTS = True


# noinspection PyMethodMayBeStatic
class ImageObstructionClean:
    def __init__(self, images=None, image_paths=None, reference_image_index=None):
        self.images = images
        self.gray_images = None
        self.blurred_images = None
        self.image_paths = image_paths
        self.reference_image_index = reference_image_index
        self.reference_image = None
        self.gray_reference_image = None
        self.edges = None
        self.blurred_edges = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.45
        self.fontColor = (0, 255, 0)
        self.lineType = 1
        self.RANSAC_Threshold = 1.0
        self.CannyParams = [100, 200]
        # self.CannyParams = [50, 150]
        # self.CannyParams = [22, 38]
        self.normalizedEdges = None
        self.Transformation_Matrix_Background = None
        self.Transformation_Matrix_Obstruction = None
        self.homography_masks = None
        self.LoadImages()
        self.Routine()
    
    def getCurrentWorkingDirectory(self):
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
                self.gray_images = np.asarray([cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY) for i in range(len(self.images))])
                if self.reference_image_index is not None:
                    self.reference_image = self.images[self.reference_image_index]
                    self.gray_reference_image = self.gray_images[self.reference_image_index]
        except Exception as LoadImagesException:
            print("Exception occurred while attempting to load images. \n", LoadImagesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def GetEdges(self, img):
        try:
            edges = cv2.Canny(img, self.CannyParams[0], self.CannyParams[1])
            return edges
        except Exception as GetEdgesException:
            print("Exception occurred in GetEdges. \n", GetEdgesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        
    def FindEdges(self):
        try:
            if self.edges is not None:
                return
            if self.blurred_images is None:
                self.blurred_images = np.asarray([cv2.GaussianBlur(img, (9, 9), 0) for img in self.images])
            self.blurred_edges = np.asarray(
                [cv2.Canny(img, self.CannyParams[0], self.CannyParams[1]) for img in self.blurred_images])
            if USEBLURRED:
                self.edges = np.asarray([cv2.Canny(img, self.CannyParams[0], self.CannyParams[1]) for img in self.blurred_images])
            else:
                self.edges = np.asarray([cv2.Canny(img, self.CannyParams[0], self.CannyParams[1]) for img in self.images])
            if TESTING:
                dir = self.getCurrentWorkingDirectory()
                if not os.path.exists(dir + "\\Edges\\"):
                    os.makedirs(dir + "\\Edges\\")
                for i in range(len(self.edges)):
                    cv2.imwrite((dir + "/Edges") + "/frame{0:04d}.png".format(i),
                                self.edges[i])
            if PARAMTESTING and TESTING:
                for min_thresh in np.arange(0, 200, 10):
                    for max_thresh in np.arange(min_thresh+10, min_thresh + 200, 10):
                        edges = cv2.Canny(self.images[2], min_thresh, max_thresh)
                        cv2.putText(edges, 'min_thresh: {}'.format(min_thresh),
                                    (10, 10),
                                    self.font,
                                    self.fontScale,
                                    self.fontColor,
                                    self.lineType, cv2.LINE_AA)
                        cv2.putText(edges, 'max_thresh: {}'.format(max_thresh),
                                    (10, 20),
                                    self.font,
                                    self.fontScale,
                                    self.fontColor,
                                    self.lineType, cv2.LINE_AA)
                        dir = self.getCurrentWorkingDirectory()
                        if not os.path.exists(os.path.join(dir, "/Results/")):
                            os.makedirs(os.path.join(dir, "/Results/"))
                        cv2.imwrite("Results/Canny_Edge_Image_2.jpg", edges)
            return
        except Exception as FindEdgesException:
            print("Exception occurred while attempting to find edges. \n", FindEdgesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def GetEdgeFlow(self, previous_image, next_image, previous_points):
        try:
            nextPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImg=previous_image,
                                                                 nextImg=next_image,
                                                                 nextPts=None,
                                                                 prevPts=previous_points.astype(
                                                                     np.float32))
            reference_image_edges = self.GetEdges(self.images[self.reference_image_index])
            reference_edge_pixels = self.GetEdgePixels(row_index=1, edges=reference_image_edges)
            start_matched_pixels, end_matched_pixels = self.GetMatchedPixels(reference_edge_pixels,
                                                                             nextPoints, status)
    
            return nextPoints, status, start_matched_pixels, end_matched_pixels
        except Exception as GetEdgeFlowException:
            print("Exception occurred within GetEdgeFlowException. \n", GetEdgeFlowException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
     
    def GetDenseOpticalFlow(self, start_img, end_img, idx):
        try:
            hsv = np.zeros_like(self.images[0])
            hsv[..., 1] = 255
            flow = cv2.optflow.calcOpticalFlowSparseToDense(end_img, start_img)
    
            y_coords, x_coords = np.mgrid[0:self.images[idx].shape[0], 0:self.images[idx].shape[1]]
            coords = np.float32(np.dstack([x_coords, y_coords]))
            pixel_map = coords + flow
            inter_frame = cv2.remap(self.images[idx], pixel_map, None, cv2.INTER_LINEAR)
    
            # From OPENCV Tutorial
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            new_bgr = cv2.applyColorMap(bgr, cv2.COLORMAP_HSV)
            if TESTING:
                dir = self.getCurrentWorkingDirectory()
                if not os.path.exists(dir + "\\Results\\"):
                    os.makedirs(dir + "\\Results\\")
                cv2.imwrite("Results/DenseMotion_{}.jpg".format(idx), new_bgr)
                cv2.imwrite("Results/Dense Motion Warp {}.jpg".format(idx), inter_frame)
            return flow, bgr, new_bgr
        except Exception as GetDenseOpticalFlowException:
            print("Exception occurred within GetDenseOpticalFlow. \n", GetDenseOpticalFlowException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
     
    def GetHomography(self, source_points, destination_points):
        try:
            transformation_matrix, homography_mask = cv2.findHomography(
                srcPoints=source_points,
                dstPoints=destination_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.RANSAC_Threshold)
            return transformation_matrix, homography_mask
        except Exception as GetHomographyException:
            print("Exception occurred in GetHomography. \n", GetHomographyException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def Routine(self):
        try:
            starting_image = self.images[self.reference_image_index]
            matrix_background = []
            matrix_obstruction = []
            for i in range(len(self.images)):
                if i == self.reference_image_index:
                    continue
                reference_image_edges = self.GetEdges(img=self.images[self.reference_image_index])
                edge_pixels_reference_image = self.GetEdgePixels(row_index=1, edges=reference_image_edges)
                prev_img = self.images[self.reference_image_index]
                next_img = self.images[i]
                if SWAP_PTS:
                    prev_img = self.images[i]
                    next_img = self.images[self.reference_image_index]
                nextPoints, \
                status, \
                start_matched_pixels, \
                end_matched_pixels = self.GetEdgeFlow(previous_image=prev_img,
                                                      next_image=next_img,
                                                      previous_points=edge_pixels_reference_image)
                src_pts = start_matched_pixels
                dst_pts = end_matched_pixels
                if SWAP_PTS:
                    src_pts = end_matched_pixels
                    dst_pts = start_matched_pixels
                
                background_transformation_matrix, background_homography_mask = self.GetHomography(
                    source_points=src_pts, destination_points=dst_pts)
                
                matrix_background.append(background_transformation_matrix)
                background_homography_mask = background_homography_mask.astype(np.bool)
                background_pixels_start = start_matched_pixels[np.where(background_homography_mask), :]
                background_pixels_end = end_matched_pixels[np.where(background_homography_mask), :].astype(np.int8)
                temp_result_background_start = np.zeros_like(self.gray_images[0])
                back_cols_start = background_pixels_start[0][:, 0]
                back_rows_start = background_pixels_start[0][:, 1]

                back_cols_end = background_pixels_end[0][:, 0]
                back_rows_end = background_pixels_end[0][:, 1]
                
                temp_result_background_start[(back_rows_start, back_cols_start)] = 255
                temp_result_background_end = cv2.warpPerspective(temp_result_background_start,
                                                                 background_transformation_matrix,
                                                                 (self.gray_images[0].shape[1],
                                                                  self.gray_images[0].shape[0]))
                if TESTING:
                    dir = self.getCurrentWorkingDirectory()
                    if not os.path.exists(dir + "\\Results\\"):
                        os.makedirs(dir + "\\Results\\")
                    cv2.imwrite("Results/Background_idk_start_{}.png".format(i), temp_result_background_start)
                    cv2.imwrite("Results/Background_idk__end_{}.png".format(i), temp_result_background_end)
                temp_result_background_end = np.zeros_like(self.gray_images[0])
                temp_result_background_end[(back_rows_end, back_cols_end)] = 255
                if TESTING:
                    flow, img, color_mapped_img = self.GetDenseOpticalFlow(start_img=temp_result_background_start,
                                                                           end_img=temp_result_background_end, idx=i)

                start_matched_pixels_filtered = start_matched_pixels[np.where(~background_homography_mask), :][0]
                end_matched_pixels_filtered = end_matched_pixels[np.where(~background_homography_mask), :][0]

                src_pts = start_matched_pixels_filtered
                dst_pts = end_matched_pixels_filtered
                if SWAP_PTS:
                    src_pts = end_matched_pixels_filtered
                    dst_pts = start_matched_pixels_filtered
                    
                obstruction_transformation_matrix, \
                obstruction_homography_mask = self.GetHomography(source_points=src_pts,
                                                                 destination_points=dst_pts)

                obstruction_pixels = start_matched_pixels_filtered[np.where(obstruction_homography_mask), :]
                
                temp_result_obstruction = np.zeros(shape=self.gray_images[0].shape)
                back_cols_obstruction = obstruction_pixels[0][:, 0]
                back_rows_obstruction = obstruction_pixels[0][:, 1]
                temp_result_obstruction[(back_rows_obstruction, back_cols_obstruction)] = 255
                
                matrix_obstruction.append(obstruction_transformation_matrix)
                result1 = cv2.warpPerspective(self.images[i],
                                              background_transformation_matrix,
                                              (starting_image.shape[1], starting_image.shape[0]))
                result2 = cv2.warpPerspective(self.images[i],
                                              obstruction_transformation_matrix,
                                              (starting_image.shape[1], starting_image.shape[0]))
                if TESTING:
                    dir = self.getCurrentWorkingDirectory()
                    if not os.path.exists(dir + "\\Results\\"):
                        os.makedirs(dir + "\\Results\\")
                    cv2.imwrite("Results/Obstruction_idk_{}.png".format(i), temp_result_obstruction)
                    cv2.imwrite("Results/Results_{}.png".format(i), result1)
                    cv2.imwrite("Results/Results_2_{}.png".format(i), result2)
    
            result_background = np.zeros_like(self.images[i]).astype(np.float64)
            result_obstruction = np.zeros_like(self.images[i]).astype(np.float64)
            array_of_background = []
            array_of_background_split_channel = [[], [], []]
            array_of_obstruction = []
            array_of_obstruction_split_channel = [[], [], []]
            self.Transformation_Matrix_Background = np.asarray(matrix_background)
            self.Transformation_Matrix_Obstruction = np.asarray(matrix_obstruction)
    
            for i in range(len(self.images)):
                matrix_index = i
                if i > 2:
                    matrix_index -= 1
                temp_background = cv2.warpPerspective(self.images[i],
                                                      self.Transformation_Matrix_Background[matrix_index],
                                                      (self.images[i].shape[1], self.images[i].shape[0]))
                temp_obstruction = cv2.warpPerspective(self.images[i],
                                                       self.Transformation_Matrix_Obstruction[matrix_index],
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
                result_obstruction += (self.images[2] * 0.2 - temp_obstruction * 0.2)
                if TESTING:
                    dir = self.getCurrentWorkingDirectory()
                    if not os.path.exists(dir + "\\Results\\"):
                        os.makedirs(dir + "\\Results\\")
                    cv2.imwrite("Results/BackGround_{}.png".format(i), result_background)
                    cv2.imwrite("Results/Obstruction_{}.png".format(i), result_obstruction)
                    cv2.imwrite("Results/Final_BackGround_{}.png".format(i), result_background)
                    cv2.imwrite("Results/Final_Obstruction_{}.png".format(i), result_obstruction)
            if TESTING:
                dir = self.getCurrentWorkingDirectory()
                if not os.path.exists(dir + "\\Results\\"):
                    os.makedirs(dir + "\\Results\\")
                test = self.images[self.reference_image_index] + result_obstruction
                test2 = self.images[self.reference_image_index] - result_obstruction
                test_2_b, test_2_g, test_2_r = cv2.split(test2)
                test4 = np.asarray([test_2_b, test_2_g, test_2_r]).mean(axis=0)
                test5 = np.asarray([test_2_b, test_2_g, test_2_r]).min(axis=0)
                test6 = np.asarray([test_2_b, test_2_g, test_2_r]).max(axis=0)
                test3 = test - test2
                cv2.imwrite("Results/Final_Result_1.png", test)
                cv2.imwrite("Results/Final_Result_2.png", test2)
                cv2.imwrite("Results/Final_Result_3.png", test3)
                cv2.imwrite("Results/Final_Result_4.png", test4)
                cv2.imwrite("Results/Final_Result_5.png", test5)
                cv2.imwrite("Results/Final_Result_6.png", test6)
            min_b_channel = np.asarray(array_of_background_split_channel[0]).min(axis=0)
            min_g_channel = np.asarray(array_of_background_split_channel[1]).min(axis=0)
            min_r_channel = np.asarray(array_of_background_split_channel[2]).min(axis=0)
            Min_Intensity_Image = cv2.merge((min_b_channel, min_g_channel, min_r_channel))
            min_b_channel_1 = np.asarray(array_of_obstruction_split_channel[0]).min(axis=0)
            min_g_channel_1 = np.asarray(array_of_obstruction_split_channel[1]).min(axis=0)
            min_r_channel_1 = np.asarray(array_of_obstruction_split_channel[2]).min(axis=0)
            Min_Intensity_Image_Obstruction = cv2.merge((min_b_channel_1, min_g_channel_1, min_r_channel_1))
            dir = self.getCurrentWorkingDirectory()
            if not os.path.exists(dir + "\\Results\\"):
                os.makedirs(dir + "\\Results\\")
            cv2.imwrite("Results/Min_Intensity_Image_Obstruction.jpg", Min_Intensity_Image_Obstruction)
            cv2.imwrite("Results/Min_Intensity_Image.jpg", Min_Intensity_Image)
            # cv2.imwrite("Resulting_Image_RANSAC_{0:04f}.jpg".format(self.RANSAC_Threshold), Min_Intensity_Image)
            cv2.imshow("Final Result", Min_Intensity_Image)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        except Exception as RoutineException:
            print("Exception occurred while attempting to execute the Routine. \n", RoutineException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
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
    
    def GetEdgePixels(self, row_index=0, edges=None):
        try:
            if self.images is not None:
                rows, columns = np.where(edges > 0)
                if row_index == 0:
                    combined_rows_and_columns = np.column_stack((rows, columns))
                else:
                    combined_rows_and_columns = np.column_stack((columns, rows))
                return combined_rows_and_columns
        except Exception as GetEdgePixelsException:
            print("Exception occurred while attempting to execute GetEdgePixels. \n", GetEdgePixelsException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def GetMatchedPixels(self, start_pixels, end_pixels, status_array):
        try:
            start_matched = start_pixels[np.where(status_array.astype(bool)), :][0]
            end_matched = end_pixels[np.where(status_array.astype(bool)), :][0]
            return start_matched, end_matched
        except Exception as GetMatchedPixelsException:
            print("Exception occurred while attempting to execute GetMatchedPixelsException. \n",
                  GetMatchedPixelsException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
