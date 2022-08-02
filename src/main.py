import sys
import copy

import numpy as np
from math import pi#,sin,cos,asin,acos, degrees

import rospy

import moveit_commander
from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose
from utils.robot.rod_finder import rod_finder
# from utility.detect_cable    import cable_detection
from utils.robot.workspace_ctrl  import move_yumi
from utils.robot.jointspace_ctrl import joint_ctrl
from utils.robot.path_generator  import path_generator
from utils.robot.gripper_ctrl    import gripper_ctrl
from utils.robot.interpolation   import interpolation

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

# import tf
from tf.transformations import quaternion_from_matrix, quaternion_matrix

## run `roslaunch rs2pcl demo.launch` first
## the default function
def main():
    

    # bc = tf.TransformBroadcaster()

    rospy.init_node('wrap_wrap', anonymous=True)
    rate = rospy.Rate(10)
    rospy.sleep(1)

    # rospy.set_param('/robot_description_planning/joint_limits/yumi_joint_1_l/max_position', 0)

    pg = path_generator()
    gripper = gripper_ctrl()
    # goal = marker()

    ##-------------------##
    ## initializing the moveit 
    moveit_commander.roscpp_initialize(sys.argv)
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    ws_tf = workspace_tf(rate)

    ctrl_group = []
    ctrl_group.append(moveit_commander.MoveGroupCommander('left_arm'))
    ctrl_group.append(moveit_commander.MoveGroupCommander('right_arm'))
    j_ctrl = joint_ctrl(ctrl_group)

    ## initialzing the yumi motion planner
    yumi = move_yumi(robot, scene, rate, ctrl_group, j_ctrl)

    print('reset the robot pose')

    # ##-------------------##
    # ## reset the robot
    gripper.l_open()
    gripper.r_open()
    j_ctrl.robot_default_l_low()
    j_ctrl.robot_default_r_low()

    gripper.l_open()
    gripper.r_open()

    rospy.sleep(3)
    print('reset done')

    # print(ctrl_group[0].get_current_joint_values())
    # print(robot.get_current_state())

    ##-------------------##
    ## Detect the rod in the first place
    # pose_init.main(rod, ws_tf)

    key = input("Help me to put the cable on the rod! (q to quit)")
    if key =='q':
        return

    ##-------------------##
    ## generate spiral here
    step_size = 0.02
    r = rod.rod_state.r
    ## l is the parameter to be tuned
    l = 2*pi*r + 0.1
    curve_path = pg.generate_nusadua(t_rod2world, l, r, step_size)

    pg.publish_waypoints(curve_path)

    ## the arbitary value (but cannot be too arbitary) of the starting value of the last/wrist joint
    j_start_value = 2*pi-2.5

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
    j_ctrl.robot_setjoint(0, q_start0)
    rospy.sleep(2)
    q_start1 = yumi.ik_with_restrict(0, stop, j_start_value)
    print(q_start1)
    print('reach out to the rope')
    j_ctrl.robot_setjoint(0, q_start1)
    ## grabbing the rope
    gripper.l_close()

    rospy.sleep(2)

    ## wrapping
    n_samples = 10
    dt = 2
    q_knots = []
    last_j_angle = 0## wrapping

    # for i in range(len(curve_path)):
    #     print(curve_path[i])
    for i in range(len(curve_path)):
        # print('waypoint %d is: '%i, end='')
        # print(q0[0])
        last_j_angle = j_start_value - 2*pi/len(curve_path)*i
        q = yumi.ik_with_restrict(0, curve_path[i], last_j_angle)

        q_knots.append(copy.deepcopy(q))

    j_traj = interpolation(q_knots, n_samples, dt)

    print('send trajectory to actionlib')
    j_ctrl.exec(0, j_traj, 0.2)

    gripper.l_open()
    # # gripper.r_open()

    start = curve_path[-1]
    stop = copy.deepcopy(start)
    if stop.position.z > 0.1:
        stop.position.z -= 0.08
    # yumi.pose_with_restrict(0, start, last_j_angle)
    j_ctrl.robot_setjoint(0, yumi.ik_with_restrict(0, start, last_j_angle))
    rospy.sleep(2)
    # yumi.pose_with_restrict(0, stop, last_j_angle)
    j_ctrl.robot_setjoint(0, yumi.ik_with_restrict(0, stop, last_j_angle))
    j_ctrl.robot_default_l_low()

    # gripper.l_open()
    # gripper.r_open()
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'init':
            ## Initializing the environment:
            ## find the rod, and save its' information (pose, r, l) to file

            rospy.init_node('wrap_wrap', anonymous=True)
            rate = rospy.Rate(10)
            rospy.sleep(1)

            ws_tf = workspace_tf(rate)
            
            moveit_commander.roscpp_initialize(sys.argv)
            scene = moveit_commander.PlanningSceneInterface()
            rod = rod_finder(scene, rate)
            rod.find_rod(ws_tf)
    else:
        main()