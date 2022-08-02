# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
sys.path.append('../../')
import copy
import time

import numpy as np
from math import sin, cos, pi
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

import cv2
import open3d as o3d

import utils.vision.non_convex_polygon as ncp

def display_img(img):
    cv2.imshow('image', img)
    show_window = True
    while show_window:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:#ESC
            cv2.destroyAllWindows()
            show_window = False

def cluster_indices(label_num, label_list):
    # return the indices of elements of given cluster
    return np.where(label_list == label_num)[0]

class object_mask():
    def __init__(self, pc_array, img, mask_color=(0,0,0)):
        ## ================
        ## image plane
        ## NW(x0,y0)----NE(x1,y0)
        ##  |              |
        ## SW(x1,y0)----SE(x1,y1)
        self.height = img.shape[0]
        self.width = img.shape[1]

        ## ================
        ## selected point cloud
        self.x_min = 10e6
        self.x_max = -10e6
        self.y_min = 10e6
        self.y_max = -10e6
        self.z_min = 10e6
        self.z_max = -10e6

        self.img = img
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.masked_img = copy.deepcopy(self.img)
        self.pc_array = copy.deepcopy(pc_array)
        self.mask_color = mask_color

    def otsu_with_mask(self):
        ...

    def extract_by_mask(self, src, mask):
        for iy in range(self.height):
            for ix in range(self.width):
                ## Just for colorizing
                # if self.mask[iy, ix] == 1:
                #     ## selected area, keep the color
                #     # self.masked_img[iy, ix, 0] = self.mask_color[0]
                #     # self.masked_img[iy, ix, 1] = self.mask_color[1]
                #     # self.masked_img[iy, ix, 2] = self.mask_color[2]
                #     pass
                # else:
                #     ## areas taken as background
                #     self.masked_img[iy, ix, 0] = self.mask_color[0]
                #     self.masked_img[iy, ix, 1] = self.mask_color[1]
                #     self.masked_img[iy, ix, 2] = self.mask_color[2]
                #     pass
                ...
        ...

    def apply_pc_mask(self):
        ## used the point clouds of selected cluster to extract 
        ## the rod from the RGB image.
        for iy in range(self.height):
            for ix in range(self.width):
                idx = iy*self.width+ix
                ## check if the given (x,y,z) is inside the bounding box
                ## otherwise apply the mask
                p = self.pc_array[idx]

                if (p[0] > self.x_min) and (p[0] < self.x_max) and \
                   (p[1] > self.y_min) and (p[1] < self.y_max) and \
                   (p[2] > self.z_min) and (p[2] < self.z_max):
                   self.mask[iy, ix] = 1

        # kernel = np.ones((21, 21), dtype=np.uint8)
        # ## dilation
        # self.mask = cv2.dilate(self.mask, kernel, iterations=1)

    def extract_rod(self):
        ## Convert to HSV color space
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        ## Apply Gaussian filter to hue channel here?
        # blur = cv2.GaussianBlur(hsv[:,:,0],(5,5),0)
        
        feature_map = []
        for iy in range(self.height):
            for ix in range(self.width):
                if self.mask[iy, ix] == 1:
                    feature_map.append([iy, ix, hsv[iy, ix, 0]])

        feature_map = np.array(feature_map)

        # ret, th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # print(feature_map.shape)
        # plt.hist(hsv[:,:,0].reshape(-1,1), 256)
        # plt.show()

        ## use K-means or Otsu's method to get two clusters (object and background)
        ## the one with the largest area is the object (rod)
        kmeans = KMeans(n_clusters = 2).fit(feature_map[:,2].reshape(-1,1))
        centers = kmeans.cluster_centers_

        elements_in_cluster = []
        elements_in_cluster.append(cluster_indices(0, kmeans.labels_))
        elements_in_cluster.append(cluster_indices(1, kmeans.labels_))

        selected_label = 0
        if len(elements_in_cluster[0]) < len(elements_in_cluster[1]):
            selected_label = 1

        # selected_img = np.ones((self.height, self.width, 3), dtype=np.uint8)*255
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for idx, value in enumerate(feature_map):
            y = value[0]
            x = value[1]
            label = kmeans.labels_[idx]
            if label == selected_label:
                self.mask[y, x] = 255
                # selected_img[y, x, 0] = self.img[y, x, 0]
                # selected_img[y, x, 1] = self.img[y, x, 1]
                # selected_img[y, x, 2] = self.img[y, x, 2]

        kernel = np.ones((3, 3), dtype=np.uint8)
        ## dilation
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        rect = ncp.largest_rect_in_non_convex_poly(self.mask, thumbnail_size=100)

        # print(rect)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # res = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(res,[box],0,(0,0,255),2)
        # display_img(res)

        return rect
