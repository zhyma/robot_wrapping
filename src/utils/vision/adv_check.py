import cv2
import numpy as np
import copy
from math import sqrt
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
from utils.vision.rope_pre_process import find_all_contours

def helix_adv_mask(h_img, poly, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape

    sort_y = poly[poly[:,1].argsort()]
    sort_x = poly[poly[:,0].argsort()]
    y1 = sort_y[0,1]
    y2 = sort_y[-1,1]
    x1 = sort_x[0,0]
    x2 = sort_x[-1,0]

    output = np.zeros((y2-y1+1, x2-x1+1), dtype=np.uint8)

    for iy in range(0, y2-y1+1):
        for ix in range(0, x2-x1+1):
            # print("ix+x1, iy+y1: {}, {}, {}, {}".format(ix, x1, iy, y1))
            j = cv2.pointPolygonTest(poly, (int(ix+x1),int(iy+y1)), False)
            if (j>0) and (h_img[iy+y1, ix+x1] > color_range[0]) and (h_img[iy+y1, ix+x1] < color_range[1]):
                output[iy, ix] = 255
            else:
                output[iy, ix] = 0

    kernel = np.ones((3, 3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1)
    output = cv2.dilate(output, kernel, iterations=1)

    return output

def get_single_hull(img):
    cont = find_all_contours(img)
    cont = np.vstack([i for i in cont])
    hull = cv2.convexHull(cont)
    return hull

def solidity(img, prev_h, post_h):
    [height, width] = img.shape
    rope = 0
    area = 0
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(prev_h, (ix,iy), False)
            k = cv2.pointPolygonTest(post_h, (ix,iy), False)
            if (k > 0) and (j<0):
                ## within the 2nd convex hull, not in the 1st convex hull
                area += 1
                if img[iy, ix] > 100:
                    rope += 1

    return rope/area

def check_adv(img1, img2, poly, hue):
    mask1 = helix_adv_mask(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,0], poly, hue)
    mask2 = helix_adv_mask(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,0], poly, hue)

    hull1 = get_single_hull(mask1)
    hull2 = get_single_hull(mask2)

    p = solidity(mask2, hull1, hull2)

    return p
