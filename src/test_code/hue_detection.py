import cv2
import numpy as np

from scipy.stats import norm
from math import sqrt

import matplotlib.pyplot as plt

def display_img(img):
    cv2.imshow('image', img)
    show_window = True
    while show_window:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:#ESC
            cv2.destroyAllWindows()
            show_window = False

def hue_detection(img, conners, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## extract feature_map from img by using the 2d bounding box
    height = img.shape[0]
    width = img.shape[1]
    feature_map = []
    masked_image = np.zeros((height,width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(cornners, (ix,iy), False)
            if j > 0:
                feature_map.append([iy, ix, hsv[iy, ix, 0]])

    feature_map = np.array(feature_map)

    hist, _ = np.histogram(feature_map, bins=list(range(257)))
    # plt.stairs(hist, list(range(257)), fill=True)
    # plt.yscale("log")
    # plt.show()
    
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

    # plt.stairs(hist_norm, list(range(257)), fill=True)
    # x = np.linspace(0, len(hist), len(hist))
    # y = norm.pdf(x, h_idx, 3)
    # plt.plot(x, y)
    # plt.show()

    h_range = [h_idx-sigma, h_idx+sigma]
    print(h_range)
    return h_range

def img_binary(img, color_range):
    # input should be a single channel image
    height = img.shape[0]
    width = img.shape[1]
    output_img = np.zeros((height,width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            if (img[iy, ix] > color_range[0]) and (img[iy, ix] < color_range[1]):
                output_img[iy, ix] = 255
            else:
                output_img[iy, ix] = 0

    return output_img

def dog_edge(img):
    low_sigma = cv2.GaussianBlur(img, (3,3),0)
    high_sigma = cv2.GaussianBlur(img, (5,5),0)
    dog = low_sigma - high_sigma
    return dog

if __name__ == '__main__': 
    img = cv2.imread("./image1.jpg")

    cornners = np.array([[943, 378],[530,363],[533,304],[945,318]])

    rope_color = hue_detection(img, cornners)
    h_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    binary_img = img_binary(h_img, rope_color)
    # display_img(binary_img)
    dog = dog_edge(binary_img)
    display_img(dog)