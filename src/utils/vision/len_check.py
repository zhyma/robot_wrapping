import cv2
import numpy as np
import copy
from math import sqrt
from skimage.morphology import skeletonize

import sys
sys.path.append('../../')
from utils.vision.rope_pre_process import find_all_contours

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
    search = True
    node0 = node(rope_top, None)
    visit_list = [node0]
    frontier = [node0]
    ## 8 connection
    while search:
        l = len(frontier)
        ## search on all frontier nodes, 
        ## move down one level (if it's child exist),
        ## or delete the frontier node (if no child, prune greadily)
        for i in range(l-1, -1, -1):
            curr_node = frontier[i]
            for next in  [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
                x = curr_node.xy[0] + next[0]
                y = curr_node.xy[1] + next[1]
                n_children = 0
                
                visited = False
                for j in visit_list:
                    if [x,y] == j.xy:
                        visited = True

                ## search for valid kids
                if visited:
                    ## skip any visited
                    continue
                if (x < 0) or (y < 0) or (x > width-1) or (y > height-1):
                    ## skip those out of the boundary
                    continue
                if img[y, x] < 100:
                    ## skip those not being marked
                    continue
                
                ## those are the children of the current node
                n_children += 1
                new_node = node([x,y], curr_node)
                frontier.append(new_node)
                visit_list.append(new_node)

                if n_children < 1:
                    ## reach the edge of the image, does not have a child  
                    curr_node.n_children = -1
                else:
                    curr_node.n_children = n_children

            if len(frontier) > 1:
                ## more than one frontier node left, the other one must has the same length
                ## (edges between nodes are equally weighted)
                frontier.pop(i)
            else:
                ## no other frontier node left, stop searching
                search = False

    mask = None
    
    string = []
    i_node = frontier[0]
    while i_node.parent is not None:
        string = [i_node.xy] + string
        i_node = i_node.parent

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

    print("total len: {}, extra len: {}".format(frontier[0].len, extra_len))

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
