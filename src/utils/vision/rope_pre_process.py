import cv2
import numpy as np
import copy
from math import sqrt

def find_all_contours(img, size_min=50):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rope_piece = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > size_min:
            rope_piece.append(i)

    cont = [contours[i] for i in rope_piece]
    return cont

def get_subimg_coord(poly, ratio=None):
    ## extract 4:3 image that includes ropes.
    sort1 = poly[poly[:,1].argsort()]
    sort2 = poly[poly[:,0].argsort()]
    y1 = sort1[0,1]-10
    y2 = 720
    x1 = sort2[0,0]-10
    x2 = sort2[-1,0]+10

    if ratio == '4/3':
        if (x2-x1)/(y2-y1) > 640/480:
                ## too rectangle
                y1 = int(y2-(x2-x1)*480/640)
        else:
            ## too square
            xc = (x2+x1)/2
            width = (y2-y1)*640/480
            x1 = int(xc-width/2)
            x2 = int(xc+width/2)

    return ((x1,y1),(x2, y2))

def binarize_by_hue(h_img, corners, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape
    output = np.zeros((height,width), dtype=np.uint8)
    [[x1, y1], [x2, y2]] = corners
    for iy in range(y1, y2):
        for ix in range(x1, x2):
            if (h_img[iy, ix] > color_range[0]) and (h_img[iy, ix] < color_range[1]):
                output[iy, ix] = 255
            else:
                output[iy, ix] = 0

    return output

def get_subimg(img, poly):
    h_binarized = binarize_by_hue(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0],\
                                  get_subimg_coord(poly), \
                                  hue_detection(img, poly))

    cropped_corners = get_subimg_coord(poly, '4/3')
    [[x1, y1], [x2, y2]] = cropped_corners
    mask = h_binarized[y1:y2,x1:x2]

    return cropped_corners, cv2.resize(img[y1:y2,x1:x2], (640,480)), cv2.resize(mask, (640,480))

def hue_detection(img, corners, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## extract feature_map from img by using the 2d bounding box
    [height, width, channel] = img.shape
    feature_map = []
    masked_image = np.zeros((height,width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(corners, (ix,iy), False)
            if j > 0:
                feature_map.append([iy, ix, hsv[iy, ix, 0]])

    feature_map = np.array(feature_map)

    hist, _ = np.histogram(feature_map, bins=list(range(257)))

    # OTSU method to estimate the threshold
    hist_norm = hist/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(255):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    print("Threshold is %d"%thresh)

    ## find two peaks, p1: peak1, p2: peak2
    p1_val = np.amax(hist[:thresh])
    p2_val = np.amax(hist[thresh:])
    h_idx = 0
    if p1_val > p2_val:
        ## pick the one on the right
        h_idx = np.where(hist==p2_val)[0][-1]
        for i in range(thresh):
            hist[i] = 0
    else:
        ## or pick the left one
        h_idx = np.where(hist==p1_val)[0][0]
        for i in range(thresh):
            hist[i] = 0

    print("The second peak is %d"%h_idx)

    if debug:
        for iy in range(height):
            for ix in range(width):
                j = cv2.pointPolygonTest(cornners, (ix,iy), False)
                if j > 0:
                    masked_image[iy, ix] = hsv[iy, ix, 0]
                else:
                    masked_image[iy, ix] = 0

        plt.imshow(masked_image)
        plt.show()

    hist_norm = hist/hist.sum()
    sigma = 0
    for i in range(len(hist)):
        sigma += (i-h_idx)**2*hist[i]
    sigma = sqrt(sigma/hist.sum())
    print(sigma)


    h_range = [h_idx-3*sigma, h_idx+3*sigma]
    print(h_range)
    return h_range