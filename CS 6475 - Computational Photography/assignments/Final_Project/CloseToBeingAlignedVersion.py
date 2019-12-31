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

# USE FRAME 2 as the REFERENCE FRAME

PRINTALLSTUFF = True
TESTING = True


# noinspection PyMethodMayBeStatic
class ImageObstructionClean:
    np.random.seed(5)
    
    def __init__(self, images=None, image_paths=None, reference_image_index=None):
        self.images = images
        self.gray_images = None
        self.blurred_images = None
        self.image_paths = image_paths
        self.reference_image_index = reference_image_index
        self.reference_image = None
        self.gray_reference_image = None
        self.edges = None
        self.edge_pixels = []       # Just pixels that correspond to the edges found
        self.blurred_edges = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.45
        self.fontColor = (0, 255, 0)
        self.lineType = 1
        self.RANSAC_Threshold = 1.0
        # self.CannyParams = [100, 200]
        self.CannyParams = [50, 150]
        # self.CannyParams = [22, 38]
        self.normalizedEdges = None
        # self.FeatureParams = dict(maxCorners=5, qualityLevel=0.01, minDistance=7, blockSize=5)
        # self.Lk_params = dict(winSize=(15, 15), maxLevel=3,
        #                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # self.color = np.random.randint(0, 255, (self.FeatureParams["maxCorners"], 3))
        self.Transformation_Matrix = None
        self.homography_masks = None
        self.LoadImages()
        self.Routine()
        # self.TEST()
    
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
            if self.blurred_images is None:
                self.blurred_images = np.asarray([cv2.GaussianBlur(img, (9, 9), 0) for img in self.images])
            self.blurred_edges = np.asarray(
                [cv2.Canny(img, self.CannyParams[0], self.CannyParams[1]) for img in self.blurred_images])
            
            self.edges = np.asarray([cv2.Canny(img, self.CannyParams[0], self.CannyParams[1]) for img in self.images])
            if TESTING:
                for i in range(len(self.edges)):
                    cv2.imwrite((self.getCurrentWorkingDirectory() + "/Edges") + "/frame{0:04d}.png".format(i),
                                self.edges[i])
            return
        except Exception as FindEdgesException:
            print("Exception occurred while attempting to find edges. \n", FindEdgesException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def CalculateEdgeFlow(self):
        try:
            ref_edge = self.edges[self.reference_image_index]
            for i in range(len(self.images)):
                self.homography_masks = self.GetHomographyMasks()
            
            return
        except Exception as CalculateEdgeFlowException:
            print("Exception occurred while attempting to execute the CalculateEdgeFlowException. \n",
                  CalculateEdgeFlowException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def GetHomographyMasks(self):
        try:
            masks = []
            matrix = []
            for i in range(len(self.images)):
                nextPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImg=self.images[i],
                                                                     nextImg=self.reference_image,
                                                                     nextPts=None,
                                                                     prevPts=self.edge_pixels[self.reference_image_index].astype(
                                                                         np.float32))

                reference_image_matched_pixels, next_image_matched_pixels = self.GetMatchedPixels(
                    start_pixels=self.edge_pixels[self.reference_image_index],
                    end_pixels=nextPoints,
                    status_array=status)
                
                # Remember srcPoints should always be the reference image points
                transformation_matrix, mask = cv2.findHomography(
                    srcPoints=reference_image_matched_pixels,
                    dstPoints=next_image_matched_pixels,
                    method=cv2.RANSAC, ransacReprojThreshold=self.RANSAC_Threshold)
                filtered_reference_points = reference_image_matched_pixels[np.where(mask == 1), :]
                filtered_next_points = next_image_matched_pixels[np.where(mask == 1), :]
                filtered_reference_points_2 = reference_image_matched_pixels[np.where(mask != 1), :]
                filtered_next_points_2 = next_image_matched_pixels[np.where(mask != 1), :]
                
                transformation_matrix_2, mask_2 = cv2.findHomography(
                    srcPoints=filtered_reference_points_2[0],
                    dstPoints=filtered_next_points_2[0],
                    method=cv2.RANSAC, ransacReprojThreshold=self.RANSAC_Threshold)
                background_test_img_1 = np.zeros_like(self.gray_images[2])
                background_test_img_2 = np.zeros_like(self.gray_images[2])
                
                obstruction_filtered_reference_points = filtered_reference_points_2[0][np.where(mask_2 == 1), :]
                obstruction_filtered_next_points = filtered_next_points_2[0][np.where(mask_2 == 1), :]
 
                obstruction_test_img_1 = np.zeros_like(self.gray_images[2])
                obstruction_test_img_1[(obstruction_filtered_reference_points[0][:, 1], obstruction_filtered_reference_points[0][:, 0])] = 255
                obstruction_test_img_2 = np.zeros_like(self.gray_images[2])
                rows = np.abs(obstruction_filtered_next_points[0][:, 1].astype(np.int) - 15)
                cols = np.abs(obstruction_filtered_next_points[0][:, 0].astype(np.int))
                background_test_img_1[(filtered_reference_points[0][:, 1], filtered_reference_points[0][:, 0])] = 255
                background_test_img_2[(filtered_next_points[0][:, 1].astype(np.int), filtered_next_points[0][:, 0].astype(np.int))] = 255
                background_test_img_1[(obstruction_filtered_reference_points[0][:, 1], obstruction_filtered_reference_points[0][:, 0])] = 255
                background_test_img_2[(rows, cols)] = 255
                cv2.imshow("background_test_img_1", background_test_img_1)
                cv2.imshow("background_test_img_2", background_test_img_2)
                cv2.imshow("obstruction_test_img_1", obstruction_test_img_1)
                cv2.imshow("obstruction_test_img_2", obstruction_test_img_2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                matrix.append(transformation_matrix)
                masks.append(mask)
            self.Transformation_Matrix = np.asarray(matrix)
            return np.asarray(masks)
        except Exception as GetMasksException:
            print("Exception occurred within GetMasks. \n", GetMasksException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def Routine(self):
        try:
            self.FindEdges()
            self.GetEdgePixels(row_index=1)
    
            starting_image = self.images[self.reference_image_index]
            matrix = []
            for i in range(len(self.images)):
                if i == self.reference_image_index:
                    continue
                nextPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImg=self.reference_image,
                                                                     nextImg=self.images[i],
                                                                     nextPts=None,
                                                                     prevPts=self.edge_pixels[
                                                                         self.reference_image_index].astype(
                                                                         np.float32))
                start_matched_pixels = self.edge_pixels[self.reference_image_index][np.where(status.astype(bool)), :][0]
                end_matched_pixels = nextPoints[np.where(status.astype(bool)), :][0]
        
                background_transformation_matrix, background_homography_mask = cv2.findHomography(
                    srcPoints=end_matched_pixels,
                    dstPoints=start_matched_pixels,
                    method=cv2.RANSAC, ransacReprojThreshold=self.RANSAC_Threshold)
                matrix.append(background_transformation_matrix)
        
                background_homography_mask = background_homography_mask.astype(np.bool)
                background_pixels = start_matched_pixels[np.where(background_homography_mask), :]
                start_matched_pixels_filtered = start_matched_pixels[np.where(~background_homography_mask), :][0]
                end_matched_pixels_filtered = end_matched_pixels[np.where(~background_homography_mask), :][0]
                obstruction_transformation_matrix, obstruction_homography_mask = cv2.findHomography(
                    srcPoints=start_matched_pixels_filtered,
                    dstPoints=end_matched_pixels_filtered,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.RANSAC_Threshold)
        
                temp_result = np.zeros(shape=self.edges[0].shape)
                back_cols = background_pixels[0][:, 0]
                back_rows = background_pixels[0][:, 1]
                temp_result[[back_rows, back_cols]] = 255
                cv2.imwrite("Background.png", temp_result)
                result1 = cv2.warpPerspective(self.images[i],
                                              background_transformation_matrix,
                                              (starting_image.shape[1], starting_image.shape[0]))
                result2 = cv2.warpPerspective(self.images[i],
                                              obstruction_transformation_matrix,
                                              (starting_image.shape[1], starting_image.shape[0]))
                cv2.imwrite("Results_{}.png".format(i), result1)
                cv2.imwrite("Results_2_{}.png".format(i), result2)
    
            result_background = np.zeros_like(result1).astype(np.float64)
            result_obstruction = np.zeros_like(result1).astype(np.float64)
            array_of_background = []
            array_of_background_split_channel = [[], [], []]
            array_of_obstruction = []
            array_of_obstruction_split_channel = [[], [], []]
            self.Transformation_Matrix = np.asarray(matrix)
    
            for i in range(len(self.images)):
                matrix_index = i
                if i > 2:
                    matrix_index -= 1
                temp_background = cv2.warpPerspective(self.images[i], self.Transformation_Matrix[matrix_index],
                                                      (self.images[i].shape[1], self.images[i].shape[0]))
                temp_obstruction = cv2.warpPerspective(self.images[i], self.Transformation_Matrix[matrix_index],
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
                hsv = np.zeros_like(self.images[0])
                hsv[..., 1] = 255
                flow = cv2.optflow.calcOpticalFlowSparseToDense(self.images[i], self.images[self.reference_image_index])
                
                # From OPENCV Tutorial
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                new_bgr = cv2.applyColorMap(bgr, cv2.COLORMAP_HSV)
                cv2.imwrite("DenseMotion_{}.jpg".format(i), new_bgr)
                result_background += (self.images[2] * 0.2 - temp_background * 0.2)
                result_obstruction += (self.images[2] * 0.2 - temp_obstruction * 0.2)
                cv2.imwrite("BackGround_{}.png".format(i), result_background)
                cv2.imwrite("Obstruction_{}.png".format(i), result_obstruction)
                cv2.imwrite("Final_BackGround_{}.png".format(i), result_background)
                cv2.imwrite("Final_Obstruction_{}.png".format(i), result_obstruction)
            test = self.images[self.reference_image_index] + result_obstruction
            test2 = self.images[self.reference_image_index] - result_obstruction
            test_2_b, test_2_g, test_2_r = cv2.split(test2)
            test4 = np.asarray([test_2_b, test_2_g, test_2_r]).mean(axis=0)
            cv2.imwrite("Final_Result_1.png", test)
            cv2.imwrite("Final_Result_2.png", test2)
            test3 = test - test2
            cv2.imwrite("Final_Result_3.png", test3)
            cv2.imwrite("Final_Result_4.png", test4)

            min_b_channel = np.asarray(array_of_background_split_channel[0]).min(axis=0)
            min_g_channel = np.asarray(array_of_background_split_channel[1]).min(axis=0)
            min_r_channel = np.asarray(array_of_background_split_channel[2]).min(axis=0)
            Min_Intensity_Image = cv2.merge((min_b_channel, min_g_channel, min_r_channel))

            cv2.imwrite("Min_Intensity_Image.jpg", Min_Intensity_Image)
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
    
    def GetEdgePixels(self, row_index=0):
        try:
            if self.images is not None:
                for i in range(len(self.images)):
                    rows, columns = np.where(self.edges[i] > 0)
                    if row_index == 0:
                        combined_rows_and_columns = np.column_stack((rows, columns))
                    else:
                        combined_rows_and_columns = np.column_stack((columns, rows))
                    self.edge_pixels.append(combined_rows_and_columns)
            return
        except Exception as GetEdgePixelsException:
            print("Exception occurred while attempting to execute GetEdgePixels. \n", GetEdgePixelsException)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def GetTransformationMatrix(self, ref_frame=2, end_frame=0):
        try:
            if ref_frame == end_frame:
                return
            reference_frame = self.images[ref_frame]
            self.GetEdgePixels(row_index=1)

            nextPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImg=reference_frame,
                                                                 nextImg=self.images[end_frame],
                                                                 nextPts=None,
                                                                 prevPts=self.edge_pixels[ref_frame].astype(
                                                                     np.float32))

            good_old_pixels = self.edge_pixels[self.reference_image_index][np.where(status.astype(bool)), :][0]
            good_new_pixels = nextPoints[np.where(status.astype(bool)), :][0]

            start_matched_pixels = self.edge_pixels[self.reference_image_index][np.where(status.astype(bool)), :][0]
            end_matched_pixels = nextPoints[np.where(status.astype(bool)), :][0]
            print()

            background_transformation_matrix, background_homography_mask = cv2.findHomography(
                srcPoints=start_matched_pixels,
                dstPoints=end_matched_pixels,
                method=cv2.RANSAC, ransacReprojThreshold=self.RANSAC_Threshold)
            background_homography_mask = background_homography_mask.astype(np.bool)
            background_pixels = start_matched_pixels[np.where(background_homography_mask), :]
            start_matched_pixels_filtered = start_matched_pixels[np.where(~background_homography_mask), :][0]
            end_matched_pixels_filtered = end_matched_pixels[np.where(~background_homography_mask), :][0]
            obstruction_transformation_matrix, obstruction_homography_mask = cv2.findHomography(
                srcPoints=start_matched_pixels_filtered,
                dstPoints=end_matched_pixels_filtered,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.RANSAC_Threshold)
            return obstruction_transformation_matrix
        except Exception as GetTransformationMatrixException:
            print("Exception occurred while attempting to execute GetEdgePixels. \n", GetTransformationMatrixException)
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

    def TEST(self):
        try:
            return
        except Exception as TEST_Exception:
            print("Exception occurred. \n", TEST_Exception)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
