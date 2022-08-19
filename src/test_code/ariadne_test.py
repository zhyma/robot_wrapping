import cv2, os
import numpy as np
import pickle
import copy

from scipy.stats import norm
from math import sqrt

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from cv_bridge import CvBridge
# ros
import rospy
import sensor_msgs.msg
import rospkg

from ariadne_plus.srv import getSplines, getSplinesRequest, getSplinesResponse

import sys
sys.path.append('../')

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


def generateImage(img_np):
    img = Image.fromarray(img_np).convert("RGB") 
    msg = sensor_msgs.msg.Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = img.height
    msg.width = img.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * img.width
    msg.data = np.array(img).tobytes()
    return msg

def apply_mask(img, corners):
    ## extract feature_map from img by using the 2d bounding box
    height = img.shape[0]
    width = img.shape[1]
    x1 = corners[0,0]
    x2 = corners[1,0]
    y1 = corners[0,1]
    y2 = corners[2,1]
    feature_map = []
    masked_image = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(corners, (ix,iy), False)
            if j > 0:
                masked_image[iy-y1, ix-x1] = img[iy, ix]

    return masked_image

if __name__ == '__main__': 
    with open('rod_info.pickle', 'rb') as handle:
            rod_info = pickle.load(handle)
            # print(rod_info.pose)
            # print(rod_info.r)
            # print(rod_info.l)
            print(rod_info.box2d)

    img = cv2.imread('image1.jpg')
    # mask_corner= copy.deepcopy(rod_info.box2d)
    sort1 = rod_info.box2d[rod_info.box2d[:,1].argsort()]
    ## upper and lower boundry
    y1 = sort1[0,1]-10
    y2 = img.shape[0]
    sort2 = rod_info.box2d[rod_info.box2d[:,0].argsort()]
    ## left and right boundry
    x1 = sort2[0,0]-10
    x2 = sort2[-1,0]+10

    mask_corner= np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

    # mask_corner = copy.deepcopy(rod_info.box2d)
    # for i in range(4):
    #     if mask_corner[i,1] == sorted_corner[2,1] or mask_corner[i,1] == sorted_corner[3,1]:
    #         mask_corner[i,1] = height

    # print(mask_corner)
    masked = apply_mask(img, mask_corner)
    # display_img(masked)
    ## apply mask to choose the workspace

    bridge = CvBridge()
    rospy.init_node('test_ariadne_service')
    rospy.sleep(1)


    input_img = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (640,480)) # resize necessary for the network model
    img_msg = generateImage(input_img)

    rospy.wait_for_service('get_splines')
    try:
        get_cable = rospy.ServiceProxy('get_splines', getSplines)
        req = getSplinesRequest()
        req.input_image = img_msg
        resp1 = get_cable(req)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    # print("get cable:")
    print(resp1.tck)
    cv_image = bridge.imgmsg_to_cv2(resp1.mask_image, desired_encoding='passthrough')
    # cv_image = bridge.imgmsg_to_cv2(resp1.final_image, desired_encoding='passthrough')
    display_img(cv_image)

