import sys
import copy
import pickle

import rospy

import cv2

from utils.vision.rgb_camera  import image_converter
from utils.vision.rope_detect import rope_detect

import datetime


if __name__ == '__main__': 
    with open('rod_info.pickle', 'rb') as handle:
        rod_info = pickle.load(handle)

    ic = image_converter()
    rospy.init_node('ariadne_test', anonymous=True)
    rospy.sleep(1)
    while ic.has_data==False:
        print('waiting for RGB data')
        rospy.sleep(0.1)

    serial_number = 0
    rd = rope_detect(rod_info)
    rope = rd.get_rope_info()
    print(rope.hue)
    print(rope.diameter)
    # rd.gp_estimation(ic.cv_image, 100)
    # cv2.imshow('image', rd.masked_img)
    # cv2.waitKey(0)