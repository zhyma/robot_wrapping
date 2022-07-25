#!/usr/bin/env python3

import sys

from math import pi

from trac_ik_python.trac_ik import IK
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64,Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list


class joint_ctrl():

    def __init__(self, ctrl_group):
        self.ctrl_group = ctrl_group
        self.trajectory_pub = rospy.Publisher('/yumi/joint_traj_pos_controller_l/command', JointTrajectory, queue_size=10)

    def robot_reset(self):
        self.robot_default_l()
        self.robot_default_r()
        
        # # pub = rospy.Publisher('/yumi/joint_pos_controller_' + str(0+1) + '_l/command', Float64, queue_size=10)
        # pub = []
        # for i in range(7):
        #     topic_name = '/yumi/joint_pos_controller_' + str(i+1) + '_l/command'
        #     pub.append(rospy.Publisher(topic_name, Float64, queue_size=10))
        #     topic_name = '/yumi/joint_pos_controller_' + str(i+1) + '_r/command'
        #     pub.append(rospy.Publisher(topic_name, Float64, queue_size=10))

        # for i in range(7*2):
        #     if i == 1 or i == 13:
        #         # left
        #         pub[i].publish(-joints_val[i//2])
        #     else:
        #         pub[i].publish(joints_val[i//2])

        # rospy.sleep(1)

    def robot_setjoint(self, selected_group, value):
        n = selected_group
        self.ctrl_group[n].set_joint_value_target(value)
        self.ctrl_group[n].go(wait=True)
        self.ctrl_group[n].stop()

    def robot_default_r(self):
        r_joints_val = [ 1.4069, -2.0969, -0.7069, 0.2969, 0, 0, 0]
        self.ctrl_group[1].set_joint_value_target(r_joints_val)
        self.ctrl_group[1].go(wait=True)
        self.ctrl_group[1].stop()

    def robot_default_l(self):
        l_joints_val = [-1.4069, -2.0969,  0.7069, 0.2969, 0, 0, 0]
        self.ctrl_group[0].set_joint_value_target(l_joints_val)
        self.ctrl_group[0].go(wait=True)
        self.ctrl_group[0].stop()

    def robot_default_l_low(self):
        l_joints_val = [ -1.1694, -2.3213, 1.1694, -0.3665, 0.2792, 1.2392, -0.3491]
        self.ctrl_group[0].set_joint_value_target(l_joints_val)
        self.ctrl_group[0].go(wait=True)
        self.ctrl_group[0].stop()

    def robot_default_r_low(self):
        r_joints_val = [ 1.1694, -2.3213, -1.1694, -0.3665, 0.2792, 1.2392, -0.3491]
        self.ctrl_group[1].set_joint_value_target(r_joints_val)
        self.ctrl_group[1].go(wait=True)
        self.ctrl_group[1].stop()

    def pub_j_trajectory(self, group, value_list, dt, start_time=0):
        joints_str = JointTrajectory()

        joints_str.header = Header()
        joints_str.header.stamp = rospy.Time.now()
        if group == 0:
            joints_str.joint_names = ['yumi_joint_1_l','yumi_joint_2_l','yumi_joint_7_l','yumi_joint_3_l','yumi_joint_4_l','yumi_joint_5_l','yumi_joint_6_l']
        else:
            joints_str.joint_names = ['yumi_joint_1_r','yumi_joint_2_r','yumi_joint_7_r','yumi_joint_3_r','yumi_joint_4_r','yumi_joint_5_r','yumi_joint_6_r']

        for i in range(len(value_list)):
            point = JointTrajectoryPoint()
            point.positions = value_list[i]
            point.time_from_start = rospy.Duration(start_time+dt*i)
            joints_str.points.append(point)

        self.trajectory_pub.publish(joints_str)


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('yumi_test', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    ctrl_group = []
    ctrl_group.append(moveit_commander.MoveGroupCommander('left_arm'))
    ctrl_group.append(moveit_commander.MoveGroupCommander('right_arm'))


    j_ctrl = joint_ctrl(ctrl_group)
    j_ctrl.robot_reset()