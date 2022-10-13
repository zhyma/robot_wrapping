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
from utils.vision.adv_check      import check_adv
from utils.vision.len_check      import check_len

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

        self.ic = image_converter()
        while self.ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

        self.file = open("param.txt", "rw")
        self.adv = -1.0
        self.len = -1.0
        line = self.file.readline()
        if len(line) > 0:
            self.adv = float(line.split(',')[0])
            self.len = float(line.split(',')[1])


    def move2pt(self, point, j6_value, group = 0):
        q = self.yumi.ik_with_restrict(group, point, j6_value)
        if type(q) is int:
            print('No IK found')
            return -1
        else:
            self.j_ctrl.robot_setjoint(group, q)

    def move_p2p(self, start, stop, j6_value, group = 0):
        ## move from point to point with j6 value fixed
        q_start = self.yumi.ik_with_restrict(group, start, j6_value)
        if type(q_start) is int:
            print('No IK found @ start')
            return -1
        else:
            self.j_ctrl.robot_setjoint(group, q_start)
        rospy.sleep(2)
        q_stop = self.yumi.ik_with_restrict(group, stop, j6_value)
        if type(q_stop) is int:
            print('No IK found @ stop')
            return -1
        else:
            self.j_ctrl.robot_setjoint(group, q_stop)

    def pts2qs(self, pts, j_start_value, d_j):
        ## solve each workspace point to joint values
        q_knots = []
        n_pts = len(pts)
        last_j_angle = j_start_value + d_j*(n_pts-1)
        for i in range(n_pts-1, -1, -1):
            # print('point {} is\n{}\n{}'.format(i, pts[i], last_j_angle))
            q = self.yumi.ik_with_restrict(0, pts[i], last_j_angle)
            ## reversed searching, therefore use -d_j
            last_j_angle -= d_j
            if q==-1:
                ## no IK solution found, remove point
                print("No IK solution is found at point {} (out of {})".format(i, n_pts))
                # curve_path.pop(i)
            else:
                # print("point {} solved".format(i))
                q_knots.insert(0, q)

        return q_knots

    def rope_holding(self, z):
        ## go to the rope's fixed end

        ## by given the expected z=0.12 value
        ## find the grasping point along y-axis
        pos = self.rope.y_estimation(self.ic.cv_image, z, end=1)

        stop = transformation2pose(np.array([[ 1,  0, 0, pos[0]],\
                                             [ 0,  0, 1, pos[1]],\
                                             [ 0, -1, 0, pos[2]],\
                                             [ 0,  0, 0, 1    ]]))
        ## -0.04 is the offset for better capturing (opening)
        ## -0.14 is the offset for fingers (finger tip to finger belly)
        stop = pose_with_offset(stop, [-0.04, 0, -0.14])
        start = pose_with_offset(stop, [0, 0, -0.03])

        ## group=1, right
        self.move_p2p(start, stop, -0.5, group=1)
        self.gripper.r_close()
        rospy.sleep(2)

        hold = pose_with_offset(stop, [-0.03, 0, -0.03])

        print("move out of the other finger's workspace")
        self.move2pt(hold, -0.5, group=1)
        rospy.sleep(2)

    def winding(self, learning=False):
        ##---- winding task entrance here ----##
        ## reset -> load info -> wrapping step(0) -> evaluate -> repeate wrapping to evaluate 3 times -> back to starting pose
        
        self.reset()

        ## recover rod's information from the saved data
        rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            rod.info = pickle.load(handle)
            print('load rod:{}'.format(rod.info.pose))

        self.rope = rope_detect(rod.info)
        with open('rope_info.pickle', 'rb') as handle:
            self.rope.info = pickle.load(handle)
            print('load rope: with color: {}, diameter: {}'.format(self.rope.info.hue, self.rope.info.diameter))

        rod.add_to_scene()
        print('rod added to scene')

        ## get transform rod2world here
        t_rod2world = pose2transformation(rod.info.pose)
        self.ws_tf.set_tf('world', 'rod', t_rod2world)

        ##-------------------##
        ## generate spiral here, two parameters to tune: advance and l
        
        advance = 0 ## meter
        r = rod.info.r
        print("rod's radius is:{}".format(r))
        # l = 2*pi*r + 0.10
        # print('Estimated L is:{}'.format(l))
        l=0.18

        print("====starting the first wrap")
        # rod_center = copy.deepcopy(t_rod2world)
        wrapping_pose = self.rope.find_frontier(self.ic.cv_image, t_rod2world)
        t_wrapping = pose2transformation(wrapping_pose)
        # # pose = transformation2pose(t_wrapping)
        wrapping_pose.position.z = 0.15
        self.marker.show(wrapping_pose)

        # Left and right arm are not mirrored. The left hand is having some problem
        # with reaching points that are too low.
        self.rope_holding(0.12)

        ## let's do a few rounds
        for i in range(3):
            ## find the left most wrap on the rod
            self.step(t_wrapping, r, l, advance, debug = True, execute=True)


            if i > 0:
                ## get feedback
                adv_result = check_adv(self.ic.cv_image, rod.info.box2d, self.rope.info.hue, self.rope.diameter)
                len_result = check_len(self.ic.cv_image, rod.info.box2d, self.rope.info.hue)

                print("Tested advnace is: {}, extra length is: {}".format(adv_result, len_result))
            # t_wrapping = tf_with_offset(t_wrapping, [0, advance, 0])
        
        self.reset()

    def step(self, center_t, r, l, advance, debug = False, execute=True):
        curve_path = self.pg.generate_nusadua(center_t, r, l, advance)

        self.pg.publish_waypoints(curve_path)

        finger_offset = [0, 0, -0.10]


        ## the arbitary value (but cannot be too arbitary) of the starting value of the last/wrist joint
        j_start_value = 2*pi-2.5
        j_stop_value = j_start_value - 2*pi

        ## preparing the joint space trajectory
        q1_knots = []
        last_j_angle = 0.0## wrapping

        print('IK for spiral')

        ## do the ik from the last point, remove those with no solution.
        n_pts = len(curve_path)
        print("planned curve_path: {}".format(n_pts))

        d_j = -2*pi/n_pts
        print("d_j is {}".format(d_j))
        # skip the first waypoint (theta=0) and the last one (theta=2\pi)
        q1_knots = self.pts2qs(curve_path[1:-1], j_start_value+d_j, d_j)
        print("sovled q1:{}".format(len(q1_knots)))

        print("solved curve_path: {}".format(len(curve_path)))
        ## solution found, now execute
        n_samples = 10
        dt = 2
        j_traj_1 = interpolation(q1_knots, n_samples, dt)

        ## from default position move to the rope starting point
        # stop = pose_with_offset(curve_path[0], [-0.005, -0.04, 0])

        ## pt_0: entering point
        ## pt_1: grabbing point
        ## pt_2: pushback position (away from the rope, x and y directions)
        ## pt_3: pushback position (away from the rope, x direction)
        ## pt_4: pushback position (away from the rope, x direction. adjust according to current rope's pose)
        ## pt_5: pushback position (right under the rod)
        pos = self.rope.gp_estimation(self.ic.cv_image, end=0, l=l-0.02)

        ## only need the orientation of the gripper
        pt_1 = copy.deepcopy(curve_path[0])
        pt_1.position.x = pos[0]
        pt_1.position.y = pos[1]
        pt_1.position.z = pos[2]
        ## z is the offset along z-axis
        pt_1 = pose_with_offset(pt_1, finger_offset)

        # self.marker.show(curve_path[0])
        ## based on the frame of link_7 (not the frame of the rod)
        ## z pointing toward right
        pt_0 = pose_with_offset(p1, [-0.01, 0, -0.06])

        ## for straightening the rope
        pose = pose_with_offset(curve_path[-1], [0, 0.10, 0])
        line_path = self.pg.generate_line(curve_path[-2], pose)
        self.pg.publish_waypoints(curve_path + line_path)
        q2_knots = self.pts2qs(line_path, j_stop_value, 0)
        j_traj_2 = interpolation(q2_knots, 2, 0.2)

        pos = self.rope.gp_estimation(self.ic.cv_image, end=0, l=l)
        pt_2 = transformation2pose(np.array([[ 0,  1, 0, pt_0.position.x+0.04],\
                                             [ 0,  0,-1, pt_0.position.y],\
                                             [-1, 0, 0, 0.12],\
                                             [ 0,  0, 0, 1    ]]))

        pt_3 = pose_with_offset(pt_2, [0, 0, 0.06])
        pt_4 = copy.deepcopy(pt_3)
        pt_4.position.y = pos[1] - finger_offset[2]
        pt_5 = copy.deepcopy(pt_4)
        pt_5.position.x = curve_path[0].position.x

        menu  = '=========================\n'
        menu += '0. NO!\n'
        menu += '1. execute\n'
        choice = input(menu)
        if choice == 1:
            execute = True
        else:
            execute = False

        if execute:
            print('move closer to the rope')
            # self.move_p2p(pt_0, pt_1, j_start_value)
            self.move2pt(pt_0, j_start_value)
            rospy.sleep(2)
            self.move2pt(pt_1, j_start_value)
            ## grabbing the rope
            self.gripper.l_close()
            rospy.sleep(2)

            # ## test the first two points of the curve
            # self.j_ctrl.exec(0, [j_traj[0], j_traj[1]], 0.2)

            print('wrapping...')
            self.j_ctrl.exec(0, j_traj_1, 0.2)

            print('straighten out the rope')
            ## straighten out the rope
            # self.marker.show(pt_2)
            self.j_ctrl.exec(0, j_traj_2, 0.2)

            rospy.sleep(2)
            self.gripper.l_open()

            print('move out from the grasping pose')
            ## left grippermove to the side
            self.move2pt(pt_0, j_stop_value)
            rospy.sleep(2)

            print('push the rope back a little bit')
            self.move2pt(pt_2, j_stop_value + pi/2)
            rospy.sleep(2)
            
            # self.move_p2p(pt_3, pt_4, j_stop_value + pi/2)
            self.move2pt(pt_3, j_stop_value + pi/2)
            rospy.sleep(2)
            self.move2pt(pt_4, j_stop_value + pi/2)
            rospy.sleep(2)
            # self.move_p2p(pt_5, pt_2, j_stop_value + pi/2)
            self.move2pt(pt_5, j_stop_value + pi/2)
            rospy.sleep(2)
            self.move2pt(pt_2, j_stop_value + pi/2)

            print('move out of the view')
            ## left grippermove to the side
            # self.move2pt(pt_0, j_stop_value)
            self.move2pt(pt_0, j_start_value)
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
    menu += '1. reset the robot\n'
    menu += '2. winding\n'
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
