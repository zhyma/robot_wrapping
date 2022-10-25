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

    rope = rope_detect(rod_info)
    rope.get_rope_info()

    with open('rope_info.pickle', 'wb') as handle:
        pickle.dump(rope.info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Rope's information has been saved to file.")

if __name__ == '__main__':
    run = True
    menu  = '=========================\n'
    menu += '1' + '. reset the robot\n'
    menu += '2' + '. get rod info\n'
    menu += '3' + '. show rod info\n'
    menu += '4' + '. get rope info\n'
    menu += '5' + '. show rope info\n'
    # menu += '6' + '. winding\n'
    # menu += '7' + '. ##demo current parameters\n'
    menu += '0. exit\n'
    menu += 'Your input: '
    while run:
        choice = input(menu)
        if choice in ['3', '5']:
            if choice == '3':
                with open('rod_info.pickle', 'rb') as handle:
                    rod_info = pickle.load(handle)
                    print(rod_info.pose)
                    print(rod_info.r)
                    print(rod_info.l)
                    print(rod_info.box2d)
            elif choice == '5':
                with open('rope_info.pickle', 'rb') as handle:
                    rope_info = pickle.load(handle)
                    print(rope_info.hue)
                    print(rope_info.diameter)
        elif choice in ['1', '2', '4', '6', '7']:
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
                elif choice == '6':
                    ## tune the parameters with 3 wraps
                    # rw.winding()
                    ...
                elif choice == '7':
                    ##
                    ...
        else:
            ## exit
            run = False
