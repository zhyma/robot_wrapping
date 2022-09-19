import cv2
import numpy as np
import copy
from math import sqrt

from skimage.morphology import skeletonize
import operator

import sys
sys.path.append('../../')
from utils.vision.bfs import bfs

def expand_mask(shape, crop_corners, mask):
    ## shape: [heigh/y, width/x]
    ## crop_corners: ((x1, y1), (x2, y2))
    expanded = np.zeros(shape,dtype=np.uint8)

    [[x1, y1], [x2, y2]] = crop_corners
    resized_mask = cv2.resize(mask, (x2-x1,y2-y1))

    for iy in range(y2-y1):
        for ix in range(x2-x1):
            if resized_mask[iy,ix] > 100:
                expanded[iy+y1, ix+x1] = 255

    return expanded

def rope_grow(rope_seed, feature_mask):
    ## rope_seed is your dl network output
    ## feature_mask can be obtained from hue channel
    [height, width] = feature_mask.shape

    mixed = copy.deepcopy(rope_seed)

    kernel = np.ones((5, 5), np.uint8)
    feature_mask = cv2.dilate(feature_mask, kernel, iterations=1)
    feature_mask = cv2.erode(feature_mask, kernel, iterations=1)

    for iy in range(height-1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_mask[iy+1, ix] > 100:
                    mixed[iy+1, ix] = 255

    for iy in range(height-1, 0, -1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_mask[iy-1, ix] > 100:
                    mixed[iy-1, ix] = 255

    return mixed

def get_rope_mask(shape, crop_corners, dl_mask, feature_mask):
    rope_mask = rope_grow(dl_mask, feature_mask)

    flesh = np.where(rope_mask>100, 1, 0)
    skeleton = skeletonize(flesh)
    ropes = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    full_mask = expand_mask(shape, crop_corners, ropes)

    return full_mask

def find_ropes(img):
    
    ropes = []
    [height, width] = img.shape

    skip_list = []

    ## search from bottom to top for the 
    for iy in range(height-1, 0, -1):
        for ix in range(width-1, 0, -1):
            if (img[iy, ix] > 100) and ([ix, iy] not in skip_list):
                r, visited_list = bfs(img, [ix,iy], [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0]], skip_list)
                if len(r) > 1:
                    r.reverse()
                    ropes.append(rope_info(r))
                skip_list += visited_list

    ropes.sort(key=operator.attrgetter('len'), reverse=True)
    if len(ropes) >= 2:
        ## always return the one for winding first, the "fix end" as the 2nd
        if ropes[0].center[0] > ropes[1].center[0]:
            detected = [ropes[0], ropes[1]]
        else:
            detected = [ropes[1], ropes[0]]
    else:
        detected = None

    return detected

def find_gp(rope, poly, l_expect):
    ## check the distance from the bottom of the rod to the top of the link
    sort1 = poly[poly[:,1].argsort()]
    [x1, y1] = sort1[2]
    [x2, y2] = sort1[3]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    l_actual = 0

    cnt = 0
    [x, y] = rope.link[0]
    y_ = a*x+b
    while cnt < rope.len-1:
        [x, y] = rope.link[cnt]
        y_ = a*x+b
        if y > y_:
            break
        else:
            cnt += 1

    ## no intersection, estimate the distance from the 2nd link to the rod
    if (cnt==0) and y > y_:
        cnt = 1
        [x, y] = rope.link[cnt]
        l_actual = abs(-a*x+y-b)/sqrt(a**2+1)
        print('l_actual init: {0}'.format(l_actual))

    while (cnt < rope.len-1):
        link0 = rope.link[cnt-1]
        link1 = rope.link[cnt]
        l_actual += sqrt((link0[0]-link1[0])**2+(link0[1]-link1[1])**2)
        if l_actual >= l_expect:
            break
        else:
            cnt += 1

    if l_actual < l_expect:
        ## need to extend 
        [x1, y1] = rope.link[rope.len//2]
        [x2, y2] = rope.link[-1]
        l = sqrt((x1-x2)**2+(y1-y2)**2)
        dl = l_expect-l_actual
        dy = dl/l*(y2-y1)
        dx = sqrt(dl**2-dy**2)
        if x1 > x2:
            x_ = x2 + dx
        else:
            x_ = x2 - dx
        return [int(x), int(y2+dy)]
    else:
        return rope.link[cnt]

class rope_info():
    def __init__(self, rope):
        self.link = rope # [[x0, y0], [x1, y1], ...]
        self.len = len(rope)
        sum = [0, 0]
        for i in self.link:
            sum[0] += i[0]
            sum[1] += i[1]

        self.center = [sum[0]/self.len, sum[1]/self.len]