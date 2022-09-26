import sys
import copy
import pickle

import numpy as np
from math import pi#,sin,cos,asin,acos, degrees

import rospy

import moveit_commander
from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose
from utils.robot.rod_finder      import rod_finder
from utils.robot.workspace_ctrl  import move_yumi
from utils.robot.jointspace_ctrl import joint_ctrl
from utils.robot.path_generator  import path_generator
from utils.robot.gripper_ctrl    import gripper_ctrl
from utils.robot.interpolation   import interpolation
from utils.robot.visualization   import marker

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

        self.marker = marker()

    def move2pt(self, point, j6_value, group = 0):
        q = self.yumi.ik_with_restrict(group, point, j6_value)
        if type(q) is int:
            print('No IK found')
        else:
            self.j_ctrl.robot_setjoint(group, q)

    def move_p2p(self, start, stop, j6_value, group = 0):
        ## move from point to point with j6 value fixed
        q_start = self.yumi.ik_with_restrict(group, start, j6_value)
        if type(q_start) is int:
            print('No IK found @ start')
        else:
            self.j_ctrl.robot_setjoint(group, q_start)
        rospy.sleep(2)
        q_stop = self.yumi.ik_with_restrict(group, stop, j6_value)
        if type(q_stop) is int:
            print('No IK found @ stop')
        else:
            self.j_ctrl.robot_setjoint(group, q_stop)

    def rope_holding(self):
        ## go to the rope
        start = transformation2pose(np.array([[ 1,  0, 0, 0.35 ],\
                                              [ 0,  0, 1, -0.26],\
                                              [ 0, -1, 0, 0.12 ],\
                                              [ 0,  0, 0, 1    ]]))

        stop = transformation2pose(np.array([[ 1,  0, 0, 0.35 ],\
                                             [ 0,  0, 1, -0.23],\
                                             [ 0, -1, 0, 0.12 ],\
                                             [ 0,  0, 0, 1    ]]))
        ## group=1, right
        self.move_p2p(start, stop, -0.5, group=1)

        self.gripper.r_close()

        rospy.sleep(2)
        hold = transformation2pose(np.array([[ 1,  0, 0, 0.32 ],\
                                             [ 0,  0, 1, -0.26],\
                                             [ 0, -1, 0, 0.12 ],\
                                             [ 0,  0, 0, 1    ]]))

        # delta = [i/100 for i in range(0, 6, 1)]
        # for ix in delta:
        #     for iy in delta:
        #         for iz in delta:
        #             point = copy.deepcopy(hold)
        #             point.position.x -= ix
        #             point.position.y -= iy
        #             point.position.z -= iz
        #             q = self.yumi.ik_with_restrict(1, point, -0.5)
        #             if type(q) is int:
        #                 # print('No IK found')
        #                 ...
        #             else:
        #                 print("With in range [{}, {}, {}]".format(0.35-ix, -0.23-iy, 0.12-iz))

        self.move2pt(hold, -0.5, group=1)
        rospy.sleep(2)

    def winding(self):
        ##---- winding task entrance here ----##
        ## reset -> load info -> wrapping step(0) -> evaluate -> repeate wrapping to evaluate 3 times -> back to starting pose

        self.reset()
        ## recover rod's information from the saved data
        rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            rod.info = pickle.load(handle)
            print('load rod:{}'.format(rod.info.pose))

        rope = rope_detect(rod.info)
        with open('rope_info.pickle', 'rb') as handle:
            rope.info = pickle.load(handle)
            print('load rope: with color: {}, diameter: {}'.format(rope.info.hue, rope.info.diameter))

        rod.add_to_scene()
        print('rod added to scene')

        ## get transform rod2world here
        t_rod2world = pose2transformation(rod.info.pose)
        self.ws_tf.set_tf('world', 'rod', t_rod2world)

        ##-------------------##
        ## generate spiral here, two parameters to tune: advance and l
        
        advance = 0.005 ## millimeter
        r = rod.info.r
        print("rod's radius is: {}".format(r))
        l = 2*pi*r + 0.10
        print('Estimated L is {}'.format(l))
        # l = 100 ## use pixel as the unit, not meters

        ## let's do a few rounds
        cnt = 0

        self.rope_holding()

        print("====starting the first wrap")
        rod_center = copy.deepcopy(t_rod2world)
        # t_wrapping = tf_with_offset(rod_center, [-0.02, -0.06, -0.02])
        t_wrapping = tf_with_offset(rod_center, [-0.00, -0.04, 0])
        # for i in range(3):
        #     center_t[i, 3] = gripper_pos[i]
        while cnt < 1:
            ## find the left most wrap on the rod
            cnt += 1
            self.step(t_wrapping, r, l, advance, debug = True, execute=False)
            # t_wrapping = tf_with_offset(t_wrapping, [0, advance, 0])
        
        # self.j_ctrl.robot_default_l_low()
        # self.reset()

    def step(self, center_t, r, l, advance, debug = False, execute=False):
        curve_path = self.pg.generate_nusadua(center_t, l, r, advance)

        self.pg.publish_waypoints(curve_path)

        ## the arbitary value (but cannot be too arbitary) of the starting value of the last/wrist joint
        j_start_value = 2*pi-2.5

        ## preparing the joint space trajectory
        n_samples = 10
        dt = 2
        q_knots = []
        last_j_angle = 0.0## wrapping

        # for i in range(len(curve_path)):
        #     print(curve_path[i])
        print('IK for spiral')

        ## do the ik from the last point, remove those with no solution.
        n_pts = len(curve_path)
        print("planned curve_path: {}".format(n_pts))
        for i in range(n_pts-1, -1, -1):
            # print('waypoint %d is: '%i, end='')
            # print(q0[0])
            last_j_angle = j_start_value - 2*pi/len(curve_path)*i
            q = self.yumi.ik_with_restrict(0, curve_path[i], last_j_angle)
            if q==-1:
                ## no IK solution found
                print("No IK solution is found at point {} (out of {})".format(i, len(curve_path)))
                curve_path.remove(i)

            else:
                q_knots.insert(0, copy.deepcopy(q))

        print("solved curve_path: {}".format(len(curve_path)))
        ## solution found, now execute
        j_traj = interpolation(q_knots, n_samples, dt)

        ## from default position move to the rope starting point
        # print(pose2transformation(curve_path[0]))
        stop = copy.deepcopy(curve_path[0])

        # stop = transformation2pose(np.array([[-1,  0,  0,  0.441],\
        #                                      [ 0,  0, -1,  0.010],\
        #                                      [ 0, -1,  0,  0.100],\
        #                                      [ 0,  0,  0,  1    ]]))

        stop = pose_with_offset(stop, [0, 0, -0.02])

        self.marker.show(stop)
        # self.move2pt(stop, j_start_value)

        ## based on the frame of link_7 (not the frame of the rod)
        ## z pointing toward right
        start = pose_with_offset(stop, [0, 0, -0.08])
        # print('starting point to grasp is:{}\n'.format(start))

        # self.marker.show(start)

        print('move closer to the rod')
        if True:
            self.move_p2p(start, stop, j_start_value)
            ## grabbing the rope
            self.gripper.l_close()
            rospy.sleep(2)

        # self.move2pt(curve_path[0], j_start_value)
        if True:
            # print('send trajectory to actionlib')
            self.j_ctrl.exec(0, j_traj, 0.2)
            rospy.sleep(2)
            ## after release the rope, continue to move down (straighten out the rope)
            self.gripper.l_open()

        start = curve_path[-1]
        # stop = pose_with_offset(start, [0, 0.08, 0])
        # stop = pose_with_offset(start, [2*r, 0.08, 0])
        stop = copy.deepcopy(start)
        stop.position.x -= 2*r
        ## ik will return -1 (no solution) if lower than 0.1
        stop.position.z = 0.1

        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, start, last_j_angle))
        # rospy.sleep(2)
        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, stop, last_j_angle))
        if execute:
            self.move_p2p(start, stop, last_j_angle)
            rospy.sleep(2)

        start = copy.deepcopy(stop)
        stop  = pose_with_offset(start, [0, 0, -0.08])

        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, start, last_j_angle))
        # rospy.sleep(2)
        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, stop, last_j_angle))
        if execute:
            self.move_p2p(start, stop, last_j_angle)
            rospy.sleep(2)

    def reset(self):
        ##-------------------##
        ## reset the robot
        print('reset the robot pose')
        self.gripper.l_open()
        self.gripper.r_open()
        self.j_ctrl.robot_default_l_low()
        self.j_ctrl.robot_default_r_low()

        self.gripper.l_open()
        self.gripper.r_open()

        rospy.sleep(3)
        print('reset done')


if __name__ == '__main__':
    run = True
    menu  = '=========================\n'
    menu += '1' + '. reset the robot\n'
    menu += '2' + '. winding\n'
    menu += '0. exit\n'
    menu += 'Your input:'
    while run:
        choice = input(menu)
        
        if choice in ['1', '2']:
            rospy.init_node('wrap_wrap', anonymous=True)
            # rate = rospy.Rate(10)
            rospy.sleep(1)

            rw = robot_winding()
            if choice == '1':
                ## reset the robot
                rw.reset()
            elif choice == '2':
                ## tune the parameters with 3 wraps
                rw.winding()

        else:
            ## exit
            run = False
