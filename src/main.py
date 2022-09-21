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
    with open('rod_info.pickle', 'rb') as handle:
        rod_info = pickle.load(handle)

    rd = rope_detect(rod_info)
    rope = rd.get_rope_info()

    with open('rope_info.pickle', 'wb') as handle:
        pickle.dump(rope, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Rope's information has been saved to file.")

if __name__ == '__main__':
    run = True
    menu  = '1' + '. reset the robot\n'
    menu += '2' + '. get rod info\n'
    menu += '3' + '. show rod info\n'
    menu += '4' + '. get rope info\n'
    menu += '5' + '. train 3 wraps\n'
    menu += '6' + '. demo current parameters\n'
    menu += '0. exit\n'
    while run:
        choice = input(menu)
        if choice == '3':
            with open('rod_info.pickle', 'rb') as handle:
                rod_info = pickle.load(handle)
                print(rod_info.pose)
                print(rod_info.r)
                print(rod_info.l)
                print(rod_info.box2d)
        elif choice in ['1', '2', '4', '5', '6']:
            rospy.init_node('wrap_wrap', anonymous=True)
            # rate = rospy.Rate(10)
            rospy.sleep(1)
            if choice == '2':
                ## init rod
                rod_init()
            elif choice == '4':
                ## init rope
                rope_init()
            else:
                rw = robot_winding()
                if choice == '1':
                    ## reset the robot
                    rw.reset()
                elif choice == '5':
                    ## tune the parameters with 3 wraps
                    rw.winding()
                elif choice == '6':
                    ##
                    ...
        else:
            ## exit
            run = False
