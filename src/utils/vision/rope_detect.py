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
from utils.vision.rope_post_process import get_rope_mask, find_ropes, find_gp
from utils.vision.adv_check import helix_adv_mask
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
        self.mask = None
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
        [x1, y1] = top_edge[0]
        [x2, y2] = top_edge[1]
        a = (y2-y1)/(x2-x1)
        b = y1-a*x1

        [height, width] = mask.shape
        ## find the pixel that is at right top corner, being masked, and on the top edge of the rod
        for ix in range(x2-1, x1, -1):
            iy = int(a*ix+b)
            if mask[iy, ix] > 100:
                break

        frontier_2d = [ix, iy]
        adv = [0, self.rod_info.l * (ix-(x1+x2)/2)/(x2-x1), 0]

        rot = center_tf[:3, :3]
        new_tf = copy.deepcopy(center_tf)
        d_trans = np.dot(rot, np.array(adv))
        for i in range(3):
            new_tf[i, 3]+=d_trans[i]
            
        return new_tf


    def gp_estimation(self, img, l=100, plt_debug=False):
        ## estimating grasping point.
        ## Input: given an image and expecting length (l) of the rope (from rod to the grasping point)

        ## crop to get the workspace
        crop_corners, cropped_img, feature_mask = get_subimg(img, self.rod_info.box2d)

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

        full_mask = get_rope_mask(img.shape[:2], crop_corners, gray, feature_mask)
        r = find_ropes(full_mask)

        l_expect = l
        ## get back the grasping point
        gp = find_gp(r[0], self.rod_info.box2d, l_expect)

        self.masked_img = copy.deepcopy(img)
        if r is not None:
            for i in r[1].link:
                self.masked_img = cv2.circle(self.masked_img, (i[0], i[1]), radius=2, color=(0, 255, 0), thickness=-1)
            for i in r[0].link:
                self.masked_img = cv2.circle(self.masked_img, (i[0], i[1]), radius=2, color=(255, 0, 0), thickness=-1)

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

            self.masked_img = cv2.circle(self.masked_img, (gp[0], gp[1]), radius=5, color=(0, 0, 255), thickness=-1)
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.masked_img, encoding='passthrough'))

            # estimate distance, actual, measured in meters
            dy = dx_p * self.scale
            dz = -dy_p * self.scale

            x = self.rod_info.pose.position.x + self.rod_info.r
            y = self.rod_info.pose.position.y + dy
            z = self.rod_info.pose.position.z + dz
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