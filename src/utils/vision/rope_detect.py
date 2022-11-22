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
sys.path.append('../../')
from utils.vision.rgb_camera import image_converter
from utils.vision.rope_pre_process import get_subimg, hue_detection
from utils.vision.rope_post_process import get_rope_skeleton, find_ropes, find_gp
from utils.vision.adv_check import helix_adv_mask, get_single_hull, find_all_contours
from utils.vision.len_check import find_rope_diameter

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

class rope_info:
    def __init__(self, hue, diameter):
        self.hue = hue
        self.diameter = diameter

class rope_detect:
    def __init__(self, rod_info):
        self.pub = rospy.Publisher('grasping_point_detect', sensor_msgs.msg.Image, queue_size=10)
        self.rod_info = rod_info
        # self.mask = None
        self.masked_img = None

        ## box2d:
        ## 3----4
        ## |    |
        ## 2----1
        x3_p = rod_info.box2d[2][0]
        x4_p = rod_info.box2d[3][0]
        y3_p = rod_info.box2d[2][1]
        y4_p = rod_info.box2d[3][1]
        l_pixel = sqrt((x3_p-x4_p)**2+(y3_p-y4_p)**2)
        self.scale = rod_info.l/l_pixel
        self.bridge = CvBridge()

        self.info = None

        self.frontier_2d = None
        self.detected_pieces = None

    def get_rope_info(self):
        ic = image_converter()
        while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)
        
        rope_hue = hue_detection(ic.cv_image, self.rod_info.box2d)
        mask0, _, _ = helix_adv_mask(cv2.cvtColor(ic.cv_image, cv2.COLOR_BGR2HSV)[:,:,0], self.rod_info.box2d, rope_hue)
        rope_diameter, _ = find_rope_diameter(mask0)

        self.info = rope_info(rope_hue, rope_diameter)

    def find_frontier(self, img, center_tf):
        mask, offset, top_edge = helix_adv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0], self.rod_info.box2d, self.info.hue)

        ## the tape around the rod leads to significant reflection, which cause detection failure.
        ## only use the lower half of the mask

        [height, width] = mask.shape
        # mask = mask[height//2:-1,:]
        # offset[1] += height//2

        # cv2.imshow('image', mask)
        # show_window = True
        # cv2.waitKey(0)

        cont = find_all_contours(mask)        
        hull = get_single_hull(cont)
        rect = cv2.minAreaRect(hull)
        box = np.int0(cv2.boxPoints(rect))

        sort_x = box[box[:,0].argsort()]
        right_edge = np.array([sort_x[2], sort_x[3]])

        ## x for 2D image, and y for 3D workspace
        self.frontier_2d = [int((right_edge[0][0] + right_edge[1][0])/2) + offset[0],\
                       int((right_edge[0][1] + right_edge[1][1])/2) + offset[1]]

        xc_p = (self.rod_info.box2d[2][0] + self.rod_info.box2d[0][0])/2
        yc_p = (self.rod_info.box2d[2][1] + self.rod_info.box2d[0][1])/2

        dx_p = self.frontier_2d[0] - xc_p
        dy_p = self.frontier_2d[1] - yc_p

        self.masked_img = cv2.circle(img, (self.frontier_2d[0], self.frontier_2d[1]), radius=10, color=(0, 0, 255), thickness=-1)
        self.masked_img = cv2.polylines(self.masked_img, [self.rod_info.box2d], isClosed=True, color=(255, 255, 0), thickness=3)
        self.pub.publish(self.bridge.cv2_to_imgmsg(self.masked_img, encoding='passthrough'))

        # estimate distance, actual, measured in meters
        dy = dx_p * self.scale
        dz = 0

        new_pose = copy.deepcopy(self.rod_info.pose)
        new_pose.position.y += dy
            
        return new_pose

    def get_ropes(self, img, debug=False):
        if self.info is None:
            print('No rope information, working on one')
            self.get_rope_info()

        ## estimating grasping point.
        ## Input: given an image and expecting length (l) of the rope (from rod to the grasping point)

        ## crop to get the workspace
        crop_corners, cropped_img, feature_mask = get_subimg(img, self.rod_info.box2d, self.info.hue)

        if debug:
            cv2.imshow("input_to_ariadne", cropped_img)
            cv2.waitKey(0)

        rospy.wait_for_service('get_splines')
        try:
            get_cable = rospy.ServiceProxy('get_splines', getSplines)
            req = getSplinesRequest()
            req.input_image = generateImage(cropped_img)
            resp1 = get_cable(req)

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        cv_image = self.bridge.imgmsg_to_cv2(resp1.mask_image, desired_encoding='passthrough')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if debug:
            cv2.imshow("gray", gray)
            cv2.waitKey(0)

        if debug:
            cv2.imshow("feature_mask", feature_mask)
            cv2.waitKey(0)

        rope_skeleton = get_rope_skeleton(img.shape[:2], crop_corners, gray, feature_mask)
        r = find_ropes(rope_skeleton)

        if debug:
            cv2.imshow("rope_skeleton", rope_skeleton)
            cv2.waitKey(0)

        self.masked_img = copy.deepcopy(img)
        self.masked_img = cv2.polylines(self.masked_img, [self.rod_info.box2d], isClosed=True, color=(255, 255, 0), thickness=3)
        if r is not None:
            for i in r[1].link:
                self.masked_img = cv2.circle(self.masked_img, (i[0], i[1]), radius=2, color=(0, 255, 0), thickness=-1)
            for i in r[0].link:
                self.masked_img = cv2.circle(self.masked_img, (i[0], i[1]), radius=2, color=(255, 0, 0), thickness=-1)

        return r

    def gp_estimation(self, img, end=0, l=0.15, use_last_pieces=False):
        ## end: active end (for wrapping) 0 or passive end (for holding) 1
        ## l measured in pixels
        ## anything ends with "p" means pixel space

        print("Use last pices?: {}".format(use_last_pieces))
        
        if (not use_last_pieces) or (self.detected_pieces is None):
            ## call Ariadne+
            print("Call Ariadne+")
            self.detected_pieces = self.get_ropes(img)
        else:
            print("Use the last result to save time")
        ## otherwise use the last result to save some time

        l_expect_p = l/self.scale
        ## get back the grasping point
        if end==1:
            gp = find_gp(self.detected_pieces[1], self.rod_info.box2d, l_expect_p)
        else:
            gp = find_gp(self.detected_pieces[0], self.rod_info.box2d, l_expect_p)

        #====

        ## Assume the measure of pixels and actual objects are uniformly scaled.
        # print("phsical to pixel scale is: %f/%f=%f"%(rod_info.l, l_pixel, self.scale))


        if gp is None:
            print("No possible rope end is found")
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.masked_img, encoding='passthrough'))
            return None
        else:
            ## return estimated grasping point position
            # center of the rectangle, in pixel
            xc_p = (self.rod_info.box2d[2][0] + self.rod_info.box2d[0][0])/2
            yc_p = (self.rod_info.box2d[2][1] + self.rod_info.box2d[0][1])/2

            dx_p = gp[0] - xc_p
            dy_p = gp[1] - yc_p

            self.masked_img = cv2.circle(self.masked_img, (gp[0], gp[1]), radius=8, color=(0, 0, 255), thickness=-1)
            if (end==0) and (self.frontier_2d is not None):
                self.masked_img = cv2.circle(self.masked_img, (self.frontier_2d[0], self.frontier_2d[1]), radius=8, color=(0, 255, 255), thickness=-1)
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.masked_img, encoding='passthrough'))

            # estimate distance, actual, measured in meters
            dy = dx_p * self.scale
            dz = -dy_p * self.scale

            if end==1:
                x = self.rod_info.pose.position.x - self.rod_info.r
            else:
                x = self.rod_info.pose.position.x + self.rod_info.r
            y = self.rod_info.pose.position.y + dy
            z = self.rod_info.pose.position.z + dz
            print("found grasping point: %.3f, %.3f, %.3f"%(x, y, z))
            return [x, y, z]

    def y_estimation(self, img, z, end=0):
        ## end: active end (for wrapping) 0 or passive end (for holding) 1
        ## given z, find the corresponding y along the rope
        ## anything ends with "p" means pixel space

        # center of the rectangle, in pixel
        xc_p = (self.rod_info.box2d[2][0] + self.rod_info.box2d[0][0])/2
        yc_p = (self.rod_info.box2d[2][1] + self.rod_info.box2d[0][1])/2

        dz = z - self.rod_info.pose.position.z
        dy_p = -dz/self.scale
        y_p = int(yc_p + dy_p)

        r = self.get_ropes(img)
        gp = [-1, y_p]
        dist = 10e6
        for i in r[end].link:
            if abs(i[1]-y_p) < dist:
                dist = abs(i[1]-y_p)
                gp[0] = i[0]

        ## return estimated grasping point position
        dx_p = gp[0] - xc_p
        ## estimate distance, actual, measured in meters
        dy = dx_p * self.scale

        self.masked_img = cv2.circle(self.masked_img, (gp[0], gp[1]), radius=5, color=(0, 0, 255), thickness=-1)
        self.pub.publish(self.bridge.cv2_to_imgmsg(self.masked_img, encoding='passthrough'))

        if end==1:
            x = self.rod_info.pose.position.x - self.rod_info.r
        else:
            x = self.rod_info.pose.position.x + self.rod_info.r
        y = self.rod_info.pose.position.y + dy

        print("found grasping point: %.3f, %.3f, %.3f"%(x, y, z))
        return [x, y, z]



if __name__ == '__main__': 
    with open('rod_info.pickle', 'rb') as handle:
        rod_info = pickle.load(handle)

    # img = cv2.imread('image1.jpg')
    ic = image_converter()
    rospy.init_node('ariadne_test', anonymous=True)
    rospy.sleep(1)
    while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

    serial_number = 0
    rd = rope_detect(rod_info)
    rd.gp_estimation(ic.cv_image, 0.1)
    # for i in range(30):
    #     main(ic.cv_image, rod_info.box2d, 0.1)
    #     input("test: "+str(i))