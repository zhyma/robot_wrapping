import sys
import copy
import pickle

import rospy

import cv2

from utils.vision.rgb_camera  import image_converter
from utils.vision.rope_detect import rope_detect

import datetime

def display_img(img):
    cv2.imshow('image', img)
    show_window = True
    while show_window:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:#ESC
            cv2.destroyAllWindows()
            show_window = False

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
    cv2.imshow('image', rd.masked_img)
    cv2.waitKey(0)
    now_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    # cv2.imwrite(now_time+'_detected.png', rd.masked_img)
    # cv2.imwrite(now_time+'.png', ic.cv_image)


    # for i in range(30):
    #     main(ic.cv_image, rod_info.box2d, 0.1)
    #     input("test: "+str(i))