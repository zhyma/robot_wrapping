import sys
import copy
import pickle

import numpy as np
from math import pi#,sin,cos,asin,acos, degrees

import rospy

import moveit_commander
from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose

from utils.vision.rgb_camera import image_converter

from utils.robot.rod_finder      import rod_finder
from utils.vision.rope_detect    import rope_detect, rope_info

from winding import robot_winding

# from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

# # import tf
# from tf.transformations import quaternion_from_matrix, quaternion_matrix

## run `roslaunch rs2pcl demo.launch` first

def rod_init():
    ## Initializing the environment:
    ## find the rod, and save its' information (pose, r, l) to file
    ws_tf = workspace_tf()
    
    moveit_commander.roscpp_initialize(sys.argv)
    scene = moveit_commander.PlanningSceneInterface()
    rod = rod_finder(scene)
    rod.find_rod(ws_tf)

    with open('rod_info.pickle', 'wb') as handle:
        pickle.dump(rod.info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Rod's information has been saved to file.")
    
def rope_init():
    # rope = rope_info()
    ic = image_converter()
    while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)
    img = copy.deepcopy(ic.cv_image)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Robot winding the given rope around a given rod.')
    parser.add_argument('-info', '--info', action='store_true', help='Read "rod_info.pickle", show the saved rod\'s information.')
    parser.add_argument('-rod_init', '--rod_init', action='store_true', help='Use the Realsense to obtain the rod\'s information and save to a file.')
    parser.add_argument('-rope_init', '--rope_init', action='store_true', help='Use the Realsense to obtain the rod\'s information and save to a file.')
    parser.add_argument('-reset', '--reset', action='store_true', help='reset the robot to its\' default position.')

    args = parser.parse_args()

    if args.info:
        with open('rod_info.pickle', 'rb') as handle:
            rod_info = pickle.load(handle)
            print(rod_info.pose)
            print(rod_info.r)
            print(rod_info.l)
            print(rod_info.box2d)

    else:
        rospy.init_node('wrap_wrap', anonymous=True)
        # rate = rospy.Rate(10)
        rospy.sleep(1)

        if args.rod_init:
            rod_init()
            exit()

        if arg.rope_init:
            rope_init()
            exit()

        ## initializing the robot's motion control
        rw = robot_winding()

        if args.reset:
            ## reset robot pose (e.g., move the arms out of the camera to do the init)
            rw.reset()

        else:
            ## check if rod_info.pickle exists.
            ## start to plan
            rw.winding()