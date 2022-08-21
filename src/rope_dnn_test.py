import sys
import copy
import pickle

import rospy


from utils.vision.rgb_camera import image_converter
from utils.vision.rope_dnn        import gp_estimation



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
    gp_estimation(ic.cv_image, rod_info, 0.1, debug=True)
    # for i in range(30):
    #     main(ic.cv_image, rod_info.box2d, 0.1)
    #     input("test: "+str(i))