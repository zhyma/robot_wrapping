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

import actionlib
from control_msgs.msg import  FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class joint_ctrl():

    def __init__(self, ctrl_group):
        self.ctrl_group = ctrl_group
        # self.trajectory_pub = rospy.Publisher('/yumi/joint_traj_pos_controller_l/command', JointTrajectory, queue_size=10)
        # self.client = actionlib.SimpleActionClient('/yumi/joint_traj_pos_controller_l/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.trajectory_pub = rospy.Publisher('/yumi/joint_traj_pos_controller_both/command', JointTrajectory, queue_size=10)
        self.client = actionlib.SimpleActionClient('/yumi/joint_traj_pos_controller_both/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.client.wait_for_server()

    def robot_reset(self):
        joints_val = [-1.1694, -2.3213, 1.1694, -0.3665, 0.2792, 1.2392, -0.3491,\
                       1.1694, -2.3213, -1.1694, -0.3665, 0.2792, 1.2392, -0.3491]
        self.robot_setjoint(2, joints_val)
        

    def robot_setjoint(self, group, selected_value):
        if group == 0:
            # move left arm only
            if isinstance(selected_value, list) and len(selected_value)==7:
                l_joints_val = selected_value
                r_joints_val = self.ctrl_group.get_current_joint_values()[7:]
                value = l_joints_val+r_joints_val

        elif group == 1:
            # move right arm only
            if isinstance(selected_value, list) and len(selected_value)==7:
                l_joints_val = self.ctrl_group.get_current_joint_values()[0:7]
                r_joints_val = selected_value
                value = l_joints_val+r_joints_val

        elif group == 2:
            # both arm
            if isinstance(selected_value, list) and len(selected_value)==14:
                value = selected_value
        else:
            print("The selected moving group does not exist")
            return


        self.ctrl_group.set_joint_value_target(value)
        self.ctrl_group.go(wait=True)
        self.ctrl_group.stop()


    def robot_default_r(self):
        r_joints_val = [ 1.4069, -2.0969, -0.7069, 0.2969, 0, 0, 0]
        self.robot_setjoint(1, r_joints_val)

    def robot_default_l(self):
        l_joints_val = [-1.4069, -2.0969,  0.7069, 0.2969, 0, 0, 0]
        self.robot_setjoint(0, l_joints_val)

    def robot_default_l_low(self):
        l_joints_val = [ -1.1694, -2.3213, 1.1694, -0.3665, 0.2792, 1.2392, -0.3491]
        
        self.robot_setjoint(0, l_joints_val)

    def robot_default_r_low(self):
        r_joints_val = [ 1.1694, -2.3213, -1.1694, -0.3665, 0.2792, 1.2392, -0.3491]
        self.robot_setjoint(1, r_joints_val)

    # def pub_j_trajectory(self, group, value_list, dt, start_time=0):
    #     joints_str = JointTrajectory()

    #     joints_str.header = Header()
    #     joints_str.header.stamp = rospy.Time.now()
    #     if group == 0:
    #         joints_str.joint_names = ['yumi_joint_1_l','yumi_joint_2_l','yumi_joint_7_l','yumi_joint_3_l','yumi_joint_4_l','yumi_joint_5_l','yumi_joint_6_l']
    #     else:
    #         joints_str.joint_names = ['yumi_joint_1_r','yumi_joint_2_r','yumi_joint_7_r','yumi_joint_3_r','yumi_joint_4_r','yumi_joint_5_r','yumi_joint_6_r']

    #     for i in range(len(value_list)):
    #         point = JointTrajectoryPoint()
    #         point.positions = value_list[i]
    #         point.time_from_start = rospy.Duration(start_time+dt*i)
    #         joints_str.points.append(point)

    #     self.trajectory_pub.publish(joints_str)

    def exec(self, group, input_value, dt, start_time=0):
        goal =  FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['yumi_joint_1_l','yumi_joint_2_l','yumi_joint_7_l','yumi_joint_3_l','yumi_joint_4_l','yumi_joint_5_l','yumi_joint_6_l',\
                                       'yumi_joint_1_r','yumi_joint_2_r','yumi_joint_7_r','yumi_joint_3_r','yumi_joint_4_r','yumi_joint_5_r','yumi_joint_6_r']

        value_list = []
        if group == 0:
            # move left arm only
            if isinstance(input_value, list) and isinstance(input_value[0], list) and (len(input_value[0])==7):
                r_joints_val = self.ctrl_group.get_current_joint_values()[7:]
                for i in range(len(input_value)):
                    value_list.append(input_value[i]+r_joints_val)

        elif group == 1:
            # move right arm only
            # not being tested yet
            print("I won't be suprised if right arm is not working...Have NOT been tested yet!")
            if isinstance(input_value, list) and isinstance(input_value[0], list) and len(input_value[0])==7:
                l_joints_val = self.ctrl_group.get_current_joint_values()[0:7]
                for i in range(len(input_value)):
                    value_list.append(l_joints_val+input_value[i])

        elif group == 2:
            # both arm
            # not being tested yet
            print("I won't be suprised if both-arm mode is not working...Have NOT been tested yet!")
            if isinstance(input_value, list) and isinstance(input_value[0], list) and len(input_value[0])==14:
                value_list = input_value
        else:
            print("The selected moving group does not exist")
            return

        for i in range(len(value_list)):
            point = JointTrajectoryPoint()
            point.positions = value_list[i]
            point.time_from_start = rospy.Duration(start_time+dt*i)

            goal.trajectory.points.append(point)

        self.client.send_goal(goal)
        self.client.wait_for_result()

        return self.client.get_result()


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('yumi_test', anonymous=True)
    robot = moveit_commander.RobotCommander()

    print(robot.get_group_names())
    scene = moveit_commander.PlanningSceneInterface()

    ctrl_group = moveit_commander.MoveGroupCommander('both_arms')


    j_ctrl = joint_ctrl(ctrl_group)
    print(j_ctrl.ctrl_group.get_current_joint_values())
    # print(j_ctrl.ctrl_group.get_current_joint_values()[0:7]) #left
    # print(j_ctrl.ctrl_group.get_current_joint_values()[7:])  #right
    j_ctrl.robot_reset()