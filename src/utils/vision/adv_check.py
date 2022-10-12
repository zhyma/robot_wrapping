import cv2
import numpy as np
import copy

from math import sqrt

import matplotlib.pyplot as plt

def helix_adv_mask(h_img, poly, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape

    sort_y = poly[poly[:,1].argsort()]
    sort_x = poly[poly[:,0].argsort()]
    y1 = sort_y[0,1]
    y2 = sort_y[-1,1]
    x1 = sort_x[0,0]
    x2 = sort_x[-1,0]

    offset = [x1, y1]

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

    top_edge = np.array([[sort_y[0][0]-x1, sort_y[0][1]-y1], [sort_y[1][0]-x1, sort_y[1][1]-y1]])

    return output, offset, top_edge[top_edge[:,0].argsort()]

def find_all_contours(img, size_min=50):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rope_piece = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > size_min:
            rope_piece.append(i)

    cont = [contours[i] for i in rope_piece]
    return cont

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

def check_adv(img, poly, hue, rope_diameter):
    ## return percentage gap_area/(gap_area+rope_area)

    mask, offset, _ = helix_adv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0], poly, hue)

    ## need to know the rope diameter, and the bottom edge of the rod
    [height, width] = mask.shape

    new_mask = np.zeros((height, width), dtype=np.uint8)

    cont = find_all_contours(mask)
    cv2.fillPoly(new_mask, pts=[i for i in cont], color=255)

    # display_img(new_mask)

    img4debug = np.zeros((height, width), dtype=np.uint8)
    # ## find the right most contour
    last_rope_area = 0
    last_gap_area = 0

    for iy in range(0, height):
        new_mask[iy, width-1] = 0

        layer_rope_area = 0
        layer_gap_area = 0

        ix = width-2
        ## skip empty part
        while ix >= 1:
            if new_mask[iy, ix] < 100:
                ix-=1
                continue
            else:
                break

        ## find the last wrap, meet the first gap or two wraps are cling to each other
        cnt_n = 0 ## number of consecutive masked pixels
        while mask[iy, ix] > 100 and (ix >= 1):
            if cnt_n <= rope_diameter:
                img4debug[iy, ix] = 255
            else:
                img4debug[iy, ix] = 150
            layer_rope_area += 1
            cnt_n += 1
            ix -= 1
            ## found two clinged wraps
            if cnt_n > rope_diameter:
                layer_rope_area += rope_diameter
                layer_gap_area = -1
                break

        if layer_gap_area == -1:
            layer_gap_area = 0
        else:
            ## scan for layer gap_area
            while mask[iy, ix] < 100 and (ix >= 1):
                layer_gap_area += 1
                ix -= 1

            if ix <= 1:
                layer_gap_area = 0

        # print("layer_rope_area: {}, layer_gap_area: {}".format(layer_rope_area, layer_gap_area))


        last_rope_area += layer_rope_area
        last_gap_area  += layer_gap_area

        img4debug[iy, ix] = 255

    print("find rope area: {}, find gap area: {}, percentage: {}".format(last_rope_area, last_gap_area, last_gap_area/(last_gap_area+last_rope_area)))

    # new_img = copy.deepcopy(img)
    # for iy in range(height):
    #     for ix in range(width):
    #         if img4debug[iy, ix] > 200:
    #             new_img[iy+offset[1], ix+offset[0]][1] = 255
    #         elif img4debug[iy, ix] > 100 and img4debug[iy, ix] < 200:
    #             new_img[iy+offset[1], ix+offset[0]][2] = 255

    return last_gap_area/(last_gap_area+last_rope_area)

if __name__ == '__main__': 
    from rope_pre_process import hue_detection
    img_list = ["./quality/09-06-10-27-48.png",\
                "./quality/09-06-10-28-31.png",\
                "./quality/09-06-10-29-12.png",\
                "./quality/09-06-10-30-12.png"]

    i = 2
    img1 = cv2.imread(img_list[i])
    img2 = cv2.imread(img_list[i+1])

    poly = np.array([[923, 391],[508,370],[512,310],[927,331]])

    ## extend the size:
    rope_hue = hue_detection(img1, poly)
    mask1 = helix_adv_mask(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
    mask2 = helix_adv_mask(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
    
    fig = plt.figure(figsize=(8,8))
    ax0 = plt.subplot2grid((3,2),(0,0))
    ax1 = plt.subplot2grid((3,2),(0,1))
    ax2 = plt.subplot2grid((3,2),(1,0))
    ax3 = plt.subplot2grid((3,2),(1,1))
    ax4 = plt.subplot2grid((3,2),(2,0))
    ax5 = plt.subplot2grid((3,2),(2,1))

    ax0.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.imshow(mask1)
    ax3.imshow(mask2)

    hull1 = get_single_hull(mask1)
    hull2 = get_single_hull(mask2)

    p = solidity(mask2, hull1, hull2)
    print(p)

    hull1_img = np.zeros(mask1.shape, dtype=np.uint8)
    cv2.drawContours(hull1_img, [hull1],-1,255,1)
    hull2_img = np.zeros(mask2.shape, dtype=np.uint8)
    cv2.drawContours(hull2_img, [hull2],-1,255,1)

    ax4.imshow(hull1_img)
    ax5.imshow(hull2_img)

    plt.tight_layout()

    plt.show()