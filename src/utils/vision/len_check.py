import cv2
import numpy as np
import copy

from math import sqrt

import matplotlib.pyplot as plt

from adv_check import helix_adv_mask, find_all_contours

from skimage.morphology import skeletonize

import sys
sys.path.append('../../')
from utils.vision.bfs import bfs

def helix_len_mask(h_img, poly, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape

    sort_y = poly[poly[:,1].argsort()]
    sort_x = poly[poly[:,0].argsort()]
    y1 = sort_y[0,1]
    y2 = sort_y[-1,1]+30
    x1 = sort_x[0,0]
    x2 = sort_x[-1,0]

    offset = [x1, y1]

    output = np.zeros((y2-y1+1, x2-x1+1), dtype=np.uint8)

    for iy in range(0, y2-y1+1):
        for ix in range(0, x2-x1+1):
            # print("ix+x1, iy+y1: {}, {}, {}, {}".format(ix, x1, iy, y1))
            # j = cv2.pointPolygonTest(poly, (int(ix+x1),int(iy+y1)), False)
            j = 1
            if (j>0) and (h_img[iy+y1, ix+x1] > color_range[0]) and (h_img[iy+y1, ix+x1] < color_range[1]):
                output[iy, ix] = 255
            else:
                output[iy, ix] = 0

    kernel = np.ones((3, 3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1)
    output = cv2.dilate(output, kernel, iterations=1)

    bottom_edge = [[sort_y[-2][0]-x1, sort_y[-2][1]-y1], [sort_y[-1][0]-x1, sort_y[-1][1]-y1]]

    return output, offset, bottom_edge

def find_rope_width(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rope_piece = []
    size_max = -1
    idx_max = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= size_max:
            size_max = area
            idx_max = i

    # hull = cv2.convexHull(contours[idx_max])
    rect = cv2.minAreaRect(contours[idx_max])
    width = rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return int(width), box

def string_search(img, bottom_edge, debug=False):
    ## search for the longest string(path) within a given image
    [height, width] = img.shape

    [x1, y1] = bottom_edge[0]
    [x2, y2] = bottom_edge[1]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    rope_top = []
    ## rope position near the top edge of the rod, take as the expected position of the wrap
    for iy in range(height):
        for ix in range(width):
            if img[iy, ix] > 100:
                ## a pixel belong to a piece of rope
                rope_top = [ix, iy]
                break
        if len(rope_top) > 0:
            break

    ## find the extra length
    ## breath first search, always prune the shortest branch greedily
    ## find the intersection between the skeleton and the 
    string, _ = bfs(img, rope_top,  [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], [])

    extra_len = 0
    for [x,y] in string:
        if y - (a*x+b) > 0:
            extra_len += 1

    if debug:
        mask = np.zeros((height, width), dtype=np.uint8)
        for [x,y] in string:
            mask[y, x] = 80
            if y - (a*x+b) > 0:
                mask[y, x] = 255

        cv2.line(mask, bottom_edge[0], bottom_edge[1], 255, 2)

    print("total len: {}, extra len: {}".format(len(string), extra_len))

    return rope_top, extra_len, mask

def remove_active(img, rope_width, bottom_edge):

    ## need to know the rope_width, and the bottom edge of the rod
    [height, width] = img.shape

    new_mask = np.zeros((height, width), dtype=np.uint8)

    cont = find_all_contours(img)
    cv2.fillPoly(new_mask, pts=[i for i in cont], color=255)

    [x1, y1] = bottom_edge[0]
    [x2, y2] = bottom_edge[1]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    for ix in range(width):
        new_mask[0, ix] = 0

    for iy in range(1, height):
        new_mask[iy, width-1] = 0

        ix = width-2
        ## skip empty part
        while ix >= 1:
            if new_mask[iy, ix] < 100:
                ix-=1
                continue
            else:
                break

        ## remove the active end
        cnt_n = 0
        while (cnt_n < rope_width) and (ix >= 1):
            new_mask[iy, ix] = 0
            cnt_n += 1
            ix -= 1

        ## skip gap (if exist)
        while (ix >= 0) and new_mask[iy, ix] < 100:
            ix -= 1

        ## keep the wrap we want to examine
        cnt_n = 0
        while (cnt_n < rope_width) and (ix >= 1):
            if new_mask[iy, ix] > 100:
                cnt_n += 1
                ix -= 1
            else:
                break

        ## remove the rest
        while (ix >= 1):
            ## any of the top three pixels are marked
            ## and below the bottom edge of the rod
            j = iy - (a*ix+b)
            or_op = int(new_mask[iy-1, ix-1]) + int(new_mask[iy-1, ix]) + int(new_mask[iy-1, ix+1])
            if (j > 0) and (or_op > 100):
                ...
            else:
                new_mask[iy, ix] = 0
            ix -= 1

    ## assume the largest contours is the piece of rope we want to check (there will be a small piece)
    contours, hierarchy = cv2.findContours(new_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    size = 0
    for i in range(len(contours)):
        i_size = cv2.contourArea(contours[i])
        if i_size > size:
            size = i_size
            idx = i

    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, pts=[contours[idx]], color=255)

    flesh = np.where(output>100, 1, 0)
    skeleton = skeletonize(flesh)
    rope_skeleton = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    return rope_skeleton

class node():
    def __init__(self, xy, parent):
        self.xy = xy
        self.parent = None
        self.len = 0
        if parent is not None:
            self.parent = parent
            self.len = parent.len + 1
            
        self.n_children = 0

if __name__ == '__main__': 
    from rope_pre_process import hue_detection

    fig = plt.figure(figsize=(16,8))
    ax = []
    for i in range(3):
        for j in range(4):
            ax.append(plt.subplot2grid((3,4),(i,j)))

    img_list = ["./quality/09-06-10-27-48.png",\
                "./quality/09-06-10-28-31.png",\
                "./quality/09-06-10-29-12.png",\
                "./quality/09-06-10-30-12.png"]

    img = []
    for i in range(4):
        img.append(cv2.imread(img_list[i]))
        ax[i].imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))

    poly = np.array([[923, 391],[508,370],[512,310],[927,331]])

    ## find the rope in the first image and estimate its' property:
    
    rope_hue = hue_detection(img[0], poly)
    mask0 = helix_adv_mask(cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
    rope_width, box0 = find_rope_width(mask0)

    ax[4].imshow(mask0)
    box0_img = np.zeros(mask0.shape, dtype=np.uint8)
    cv2.drawContours(box0_img, [box0],-1,255,1)
    ax[8].imshow(box0_img)

    mask = []
    contours = []
    inter_step_img = []
    for i in range(3):
        sub_mask, offset, bottom_edge = helix_len_mask(cv2.cvtColor(img[i+1], cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
        mask.append(sub_mask)
        new_mask = remove_active(mask[i], rope_width, bottom_edge)
        top, extra_len, filtered_string = string_search(new_mask, bottom_edge, debug=True)
        filtered_string = cv2.circle(filtered_string, top, radius=2, color=255, thickness=-1)

        ax[i+5].imshow(mask[i])
        inter_step_img.append(filtered_string)
        ax[i+9].imshow(inter_step_img[i])

    plt.tight_layout()

    plt.show()