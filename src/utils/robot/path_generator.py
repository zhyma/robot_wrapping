#!/usr/bin/env python3

from math import pi, sin, cos, sqrt, atan2
import numpy as np
import sys
import copy

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from transforms3d import euler

from tf.transformations import quaternion_from_matrix, quaternion_matrix


class path_generator():

    def __init__(self):
        self.waypoint_pub = rospy.Publisher('yumi_waypoint', Path, queue_size=1, latch=True)

    def generate_nusadua(self, t_rod, r, l, advance):
        ## For left hand
        ## curve on x-z plane
        ## from 2d plane curve to world frame path: [xr, adv, yr]
        path = []
        n_samples = 12

        finger_offset = 0.04

        ## T^{ref}_{obj}: from ref to obj
        ## t_ft2gb: Finger Tip with respect to the Gripper Base
        ## t_gb2ft is the inverse matrix (gb with repesct to ft)
        t_ft2gb = np.array([[1, 0, 0, 0],\
                            [0, 1, 0, 0],\
                            [0, 0, 1, 0.125],\
                            [0, 0, 0, 1]])
        t_gb2ft = np.linalg.inv(t_ft2gb)

        a = 0

        for i in range(n_samples+1):
            
            t = (2+0)*pi/n_samples * i
            x = r*cos(t)
            z = r*sin(t)
            ## based on the world coordinate
            xr  = x-(l -a*(t) -t*r)*sin(t) * ( 1)
            zr  = z+(l -a*(t) -t*r)*cos(t) * (-1)
            # adv = advance * i/n_samples + finger_offset*(n_samples-i)/n_samples
            adv = advance * i/n_samples

            # ## Use circle to test
            # xr = (r+0.2)*sin(t)
            # yr = (r+0.2)*cos(t)
            # adv = 0

            t_curve2d = np.array([[1, 0, 0, xr ],\
                                  [0, 1, 0, adv],\
                                  [0, 0, 1, zr ],\
                                  [0, 0, 0, 1  ]])

            ## project the curve on the plane that perpendicular to the rod
            t_ft2world = np.dot(t_rod, t_curve2d)

            ## sin(theta') and cos(theta')
            st = xr/sqrt(xr**2+zr**2)
            ct = zr/sqrt(xr**2+zr**2)

            t_orientation = np.array([[ ct, st,  0, 0],\
                                      [  0,  0, -1, 0],\
                                      [-st, ct,  0, 1],\
                                      [  0,  0,  0, 1]])

            t_ft2world[:3,:3] = np.dot(t_rod, t_orientation)[:3,:3]

            ## reference frame:world. t_world2gb = t_world2ft*inv(t_gb2ft)
            t_gb2world = np.dot(t_ft2world,t_gb2ft)
            o = t_gb2world[:3,3]
            q = quaternion_from_matrix(t_gb2world)

            pose = Pose()
            pose.position.x = o[0]
            pose.position.y = o[1]
            pose.position.z = o[2]

            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            # theta = (220.0-360.0/n_samples*i)*pi/180.0
            # new_pose = step_back(pose, theta)

            path.append(pose)
        
        return path

    def generate_spiral(self, spiral_params, gripper_states):
        path = []
        # # test to plot a line.
        # for i in range(30):
        #     path.append([0.3, i/100, 0.4])

        
        # s is the center of the rod
        # x (front-back), y (left-right), z (up-down).
        # x and z are used for spiral, y is used for advance
        s = spiral_params
        # s = [0.3, 0.1, 0.4]
        # g is the gripper's starting position
        g = gripper_states
        # g = [0.6, 0.1, 0.42]
        # starting angle theta_0
        t_0 = 0.2
        # ending angle theta_end
        t_e = 0.2+pi*2
        n_samples = 12
        n = -1/2
        # offset theta_offset
        # t_os = 0
        t_os = atan2((g[2]-s[2]),(g[0]-s[0]))

        # r = a\theta^(-1/2)
        r_0 = np.sqrt((s[0]-g[0])**2+(s[2]-g[2])**2)
        a = r_0/np.power(t_0, n)
        for i in range(n_samples):
            dt = (t_e-t_0)*i/n_samples
            r = a*np.power((t_0+dt), n)
            t = -dt + t_os
            x = r*np.cos(t) + s[0]
            # x = -r*np.cos(t) + s[0]
            y = g[1] + 0.005*i
            z = r*np.sin(t) + s[2]
            path.append([x,y,z])
        
        return path

    def generate_line(self, start, stop, n_samples = 10):
        path = []

        [x0, y0, z0] = [start.position.x, start.position.y, start.position.z]
        [x1, y1, z1] = [ stop.position.x,  stop.position.y,  stop.position.z]


        for i in range(n_samples):
            pose = copy.deepcopy(stop)
            pose.position.x = x0+(x1-x0)*i/(n_samples-1)
            pose.position.y = y0+(y1-y0)*i/(n_samples-1)
            pose.position.z = z0+(z1-z0)*i/(n_samples-1)
            path.append(pose)

        return path

    def publish_waypoints(self, path):
        """
        Publish the ROS message containing the waypoints
        """

        msg = Path()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()

        for wp in path:
            pose = PoseStamped()
            pose.pose.position.x = wp.position.x
            pose.pose.position.y = wp.position.y
            pose.pose.position.z = wp.position.z

            pose.pose.orientation.x = wp.orientation.x
            pose.pose.orientation.y = wp.orientation.y
            pose.pose.orientation.z = wp.orientation.z
            pose.pose.orientation.w = wp.orientation.w
            msg.poses.append(pose)

        self.waypoint_pub.publish(msg)
        rospy.loginfo("Published {} waypoints.".format(len(msg.poses)))
        return 
