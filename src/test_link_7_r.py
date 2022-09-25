import sys
import copy
import pickle

import numpy as np
from math import pi,sin,cos,asin,acos, degrees

import rospy

import moveit_commander
from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose
from utils.robot.rod_finder      import rod_finder
# from utility.detect_cable        import cable_detection
from utils.robot.workspace_ctrl  import move_yumi
from utils.robot.jointspace_ctrl import joint_ctrl
from utils.robot.path_generator  import path_generator
from utils.robot.gripper_ctrl    import gripper_ctrl
from utils.robot.interpolation   import interpolation

from utils.vision.rgb_camera     import image_converter
from utils.vision.rope_detect    import rope_detect

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

# import tf
from tf.transformations import quaternion_from_matrix, quaternion_matrix

## run `roslaunch rs2pcl demo.launch` first

def tf_with_offset(tf, offset):
    rot = tf[:3, :3]
    new_tf = copy.deepcopy(tf)
    d_trans = np.dot(rot, np.array(offset))
    for i in range(3):
        new_tf[i, 3]+=d_trans[i]

    return new_tf

def pose_with_offset(pose, offset):
    ## the offset is given based on the current pose's translation
    tf = pose2transformation(pose)
    new_tf = tf_with_offset(tf, offset)
    new_pose = transformation2pose(new_tf)

    ## make sure it won't go too low
    if new_pose.position.z < 0.03:
            new_pose.position.z = 0.03
    return new_pose

class robot_winding():
    def __init__(self):
        self.gripper = gripper_ctrl()

        ##-------------------##
        ## initializing the moveit 
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()
        self.ws_tf = workspace_tf()

        self.ctrl_group = []
        self.ctrl_group.append(moveit_commander.MoveGroupCommander('left_arm'))
        self.ctrl_group.append(moveit_commander.MoveGroupCommander('right_arm'))
        self.j_ctrl = joint_ctrl(self.ctrl_group)
        ## initialzing the yumi motion planner
        self.yumi = move_yumi(self.robot, self.scene, self.ctrl_group, self.j_ctrl)
        self.pg = path_generator()


    def joint_test(self):

        rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            rod.info = pickle.load(handle)
            print('load rod:{}'.format(rod.info.pose))

        rod.add_to_scene()
        print('rod added to scene')

        ## get transform rod2world here

        ht_holding = np.array([[ 1,  0, 0, 0.36 ],\
                               [ 0,  0, 1, -0.27],\
                               [ 0, -1, 0, 0.12 ],\
                               [ 0,  0, 0, 1    ]])
        pose_holding = transformation2pose(ht_holding)

        print(pose_holding)

        self.ws_tf.set_tf('world', 'test_target', ht_holding)

        new_tf = self.ws_tf.get_tf('world', 'yumi_link_7_r')
        print('current link 7 tf is: {}'.format(new_tf))


        jvs = self.yumi.ctrl_group[1].get_current_joint_values()
        print(jvs)
        # print(sin(jvs[-1]))
        # print(cos(jvs[-1]))
        # ht_7r = self.ws_tf.get_tf('yumi_link_6_r', 'yumi_link_7_r')
        # print(ht_7r)

        # ht_7r = self.ws_tf.get_tf('world', 'yumi_link_7_r')

        # theta = jvs[-1]

        # c = cos(theta)
        # s = sin(theta)

        # ## homogeneous transformation matrix from link_6_r to link_7_r
        # ht = np.array([[ c, -s, 0, 0.027],\
        #                [ 0,  0, 1, 0.029],\
        #                [-s, -c, 0, 0    ],\
        #                [ 0,  0, 0, 1    ]])

        # print(ht)
        # inv_ht = np.linalg.inv(ht)
        
        # # pose_6r = transformation2pose(np.dot(ht_7r, inv_ht))
        # self.ws_tf.set_tf('world', 'test_link_7_r', np.dot(ht_7r, inv_ht))


        # self.yumi.goto_pose(self.ctrl_group[1], pose_holding)
        q = self.yumi.ik_with_restrict(1, pose_holding, -0.50)#2*pi-2.5)

        if type(q) is int:
            print('No IK found')
        else:
            self.j_ctrl.robot_setjoint(1, q)

    
if __name__ == '__main__':

    rospy.init_node('wrap_wrap', anonymous=True)
    # rate = rospy.Rate(10)
    rospy.sleep(1)

    ## initializing the robot's motion control
    rw = robot_winding()

    rw.joint_test()