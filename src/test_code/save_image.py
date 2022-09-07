# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
import rospy
import numpy as np
from math import sin, cos, pi

import time

import copy

from sensor_msgs.msg import Image
import datetime

from cv_bridge import CvBridge, CvBridgeError
import cv2

#from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromOpen3dToRos

bridge = CvBridge()

# class rs2pc():
#     def __init__(self):
#         self.is_data_updated = False
#         # self.img_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.img_callback)
#         self.img_sub = rospy.Subscriber("/front_cam/color/image_raw", Image, self.img_callback)
#         self.image = None

#     def img_callback(self, data):
#         try:
#           cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#           self.image = copy.deepcopy(cv_image)
#         except CvBridgeError as e:
#           print(e)

#         self.is_data_updated = True

#     def cam_info_callback(self, data):
#         if self.is_k_empty:
#             for i in range(9):
#                 self.k[i] = data.K[i]
#             self.is_k_empty = False

def image_callback(msg):
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError:
        # print(e)
        ...
    else:
        now_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
        cv2.imwrite(now_time+'.png', cv2_img)



def main():
    # rs = rs2pc()

    rospy.init_node('rs2icp', anonymous=True)
    rospy.sleep(1)

    rospy.Subscriber("/front_cam/color/image_raw", Image, image_callback)


    rospy.sleep(1)
    # while rs.is_data_updated==False:
    #     rospy.spin()


if __name__ == '__main__':
    
    main()