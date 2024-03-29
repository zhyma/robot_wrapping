import sys
import copy
import pickle

import numpy as np
from math import pi#,sin,cos,asin,acos, degrees

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

    def move_p2p(self, start, stop, j6_value):
        ## move from point to point with j6 value fixed
        q_start = self.yumi.ik_with_restrict(0, start, j6_value)
        self.j_ctrl.robot_setjoint(0, q_start)
        rospy.sleep(2)
        q_stop = self.yumi.ik_with_restrict(0, stop, j6_value)
        print('reach out to the rope')
        self.j_ctrl.robot_setjoint(0, q_stop)

    def move2grasp(self):
        ## reset -> load info -> winding step() by step() -> back to starting pose

        ic = image_converter()
        while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

        print("rgb_data_ready")

        self.reset()
        ## recover rod's information from the saved data
        rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            rod.info = pickle.load(handle)
            print(rod.info.pose)

        rod.add_to_scene()
        print('rod added to scene')

        ## get transform rod2world here
        t_rod2world = pose2transformation(rod.info.pose)
        self.ws_tf.set_tf('world', 'rod', t_rod2world)

        ##-------------------##
        ## generate spiral here
        center_t = copy.deepcopy(t_rod2world)
        step_size = 0.005
        r = rod.info.r
        ## l is the parameter to be tuned
        l = 2*pi*r + 0.05

        ## let's do a few rounds
        cnt = 0

        # ## TODO replace here with the center of the rope that is on the rod
        # ## take that as the starting point of winding.
        # gripper_pos = None
        # while gripper_pos is None:
        #     img = copy.deepcopy(ic.cv_image)
        #     gripper_pos = gp_estimation(img, rod.info, l)
        #     if gripper_pos is None:
        #         input("grasping point not found, try to move the cable a little bit.")
        #     else:

        #         break

        print("====starting the first wrap")
        # for i in range(3):
        #     center_t[i, 3] = gripper_pos[i]
        while cnt < 1:
            ## grasping position
            img = copy.deepcopy(ic.cv_image)
            print('expecting l is: %.3f'%(l))
            gripper_pos = gp_estimation(img, rod.info, l)
            if gripper_pos is None:
                ## One possibility of having no result: l is too long
                input("grasping point not found, try to move the cable a little bit.")
            else:
                cnt += 1
                print(gripper_pos)
                self.step(center_t, r, l, step_size, gripper_pos)
                center_t = tf_with_offset(center_t, [0, step_size, 0])
        
        self.j_ctrl.robot_default_l_low()

    def step(self, center_t, r, l, step_size, gripper_pos):
        curve_path = self.pg.generate_nusadua(center_t, l, r, step_size)

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
        for i in range(len(curve_path)):
            # print('waypoint %d is: '%i, end='')
            # print(q0[0])
            last_j_angle = j_start_value - 2*pi/len(curve_path)*i
            q = self.yumi.ik_with_restrict(0, curve_path[i], last_j_angle)
            if q==-1:
                ## no IK solution found
                print("No IK solution is found")
                return

            q_knots.append(copy.deepcopy(q))

        ## solution found, now execute
        j_traj = interpolation(q_knots, n_samples, dt)

        ## from default position move to the rope starting point
        stop  = copy.deepcopy(curve_path[0])
        ## update the [x,y,z] only to catch the rope
        stop.position.x = gripper_pos[0]
        stop.position.y = gripper_pos[1] + 0.09
        stop.position.z = gripper_pos[2]
        ## based on the frame of link_7 (not the frame of the rod)
        ## z pointing toward right
        start = pose_with_offset(stop, [0, 0, -0.08])

        print('move closer to the rod')
        # q_start0 = self.yumi.ik_with_restrict(0, start, j_start_value)
        # self.j_ctrl.robot_setjoint(0, q_start0)
        # rospy.sleep(2)
        # q_start1 = self.yumi.ik_with_restrict(0, stop, j_start_value)
        # print('reach out to the rope')
        # self.j_ctrl.robot_setjoint(0, q_start1)
        self.move_p2p(start, stop, j_start_value)
        ## grabbing the rope
        # self.gripper.l_close()

        rospy.sleep(2)

        # print('send trajectory to actionlib')
        self.j_ctrl.exec(0, j_traj, 0.2)

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
        self.move_p2p(start, stop, last_j_angle)
        rospy.sleep(2)

        start = copy.deepcopy(stop)
        stop  = pose_with_offset(start, [0, 0, -0.08])

        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, start, last_j_angle))
        # rospy.sleep(2)
        # self.j_ctrl.robot_setjoint(0, self.yumi.ik_with_restrict(0, stop, last_j_angle))
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

def env_init():
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
    
if __name__ == '__main__':

    rospy.init_node('wrap_wrap', anonymous=True)
    # rate = rospy.Rate(10)
    rospy.sleep(1)

    ## initializing the robot's motion control
    rw = robot_winding()

    rw.move2grasp()