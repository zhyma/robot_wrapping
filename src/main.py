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

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

# import tf
from tf.transformations import quaternion_from_matrix, quaternion_matrix

## run `roslaunch rs2pcl demo.launch` first

class robot_wrap():
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

    def wrap(self):
        rate = rospy.Rate(10)
        pg = path_generator()
        self.reset()

        ## initialzing the yumi motion planner
        yumi = move_yumi(self.robot, self.scene, rate, self.ctrl_group, self.j_ctrl)

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
        step_size = 0.02
        r = rod.info.r
        ## l is the parameter to be tuned
        l = 2*pi*r + 0.1
        curve_path = pg.generate_nusadua(t_rod2world, l, r, step_size)

        pg.publish_waypoints(curve_path)

        ## the arbitary value (but cannot be too arbitary) of the starting value of the last/wrist joint
        j_start_value = 2*pi-2.5

        ## preparing the joint space trajectory
        n_samples = 10
        dt = 2
        q_knots = []
        last_j_angle = 0## wrapping

        # for i in range(len(curve_path)):
        #     print(curve_path[i])
        print('IK for spiral')
        for i in range(len(curve_path)):
            # print('waypoint %d is: '%i, end='')
            # print(q0[0])
            last_j_angle = j_start_value - 2*pi/len(curve_path)*i
            q = yumi.ik_with_restrict(0, curve_path[i], last_j_angle)
            if q==-1:
                ## no IK solution found
                print("No IK solution is found")
                return

            q_knots.append(copy.deepcopy(q))

        ## solution found, now execute
        j_traj = interpolation(q_knots, n_samples, dt)

        ## from default position move to the rope starting point
        stop  = curve_path[0]
        ht_stop = pose2transformation(stop)
        start_offset = np.array([[1, 0, 0, 0],\
                                 [0, 1, 0, 0],\
                                 [0, 0, 1, -0.08],\
                                 [0, 0, 0, 1]])
        start = transformation2pose(np.dot(ht_stop, start_offset))
        print('move closer to the rod')
        q_start0 = yumi.ik_with_restrict(0, start, j_start_value)
        print(q_start0)
        self.j_ctrl.robot_setjoint(0, q_start0)
        rospy.sleep(2)
        q_start1 = yumi.ik_with_restrict(0, stop, j_start_value)
        print(q_start1)
        print('reach out to the rope')
        self.j_ctrl.robot_setjoint(0, q_start1)
        ## grabbing the rope
        self.gripper.l_close()

        rospy.sleep(2)

        print('send trajectory to actionlib')
        self.j_ctrl.exec(0, j_traj, 0.2)

        self.gripper.l_open()

        start = curve_path[-1]
        stop = copy.deepcopy(start)
        if stop.position.z > 0.1:
            stop.position.z -= 0.08
        self.j_ctrl.robot_setjoint(0, yumi.ik_with_restrict(0, start, last_j_angle))
        rospy.sleep(2)
        self.j_ctrl.robot_setjoint(0, yumi.ik_with_restrict(0, stop, last_j_angle))
        self.j_ctrl.robot_default_l_low()

        # gripper.l_open()
        # gripper.r_open()

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
    if len(sys.argv) > 1 and sys.argv[1] == 'rod_info':
        with open('rod_info.pickle', 'rb') as handle:
            rod_info = pickle.load(handle)
            print(rod_info.pose)
            print(rod_info.r)
            print(rod_info.l)
    else:
        rospy.init_node('wrap_wrap', anonymous=True)
        rate = rospy.Rate(10)
        rospy.sleep(1)

        if len(sys.argv) > 1 and sys.argv[1] == 'init':
            env_init()
            exit()

        rw = robot_wrap()

        if len(sys.argv) > 1:
            if sys.argv[1] == 'reset':
                ## reset robot pose (e.g., move the arms out of the camera to do the init)
                rw.reset()
            else:
                print("no such an argument")
        else:
            ## check if rod_info.pickle exists.
            ## start to plan
            rw.wrap()