# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
import copy
import time

import numpy as np
from math import sin, cos, pi

import rospy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

import open3d as o3d

## convert RealSense depth data to Open3D point cloud

class rs2o3d():
    def __init__(self):
        self.is_k_empty = True
        self.is_data_updated = False
        self.k = [0]*9 # camera's intrinsic parameters
        self.cam_sub = rospy.Subscriber("/front_cam/depth/camera_info", CameraInfo, self.cam_info_callback)
        # self.img_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.img_callback)
        ## need parameter `align_depth:=true`
        self.img_sub = rospy.Subscriber("/front_cam/aligned_depth_to_color/image_raw", Image, self.img_callback)
        self.pcd = o3d.geometry.PointCloud()

        self.height = -1
        self.width = -1

        self.pub = rospy.Publisher('/rs_point_cloud', PointCloud2, queue_size=1000)
        self.FIELDS_XYZ = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                           PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                           PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),]


    def img_callback(self, data):
        if self.is_k_empty == False:
            self.height = data.height
            self.width = data.width

            np_cloud = np.zeros((self.height*self.width,3))
            # print(self.k)
            for iy in range(self.height):
                for ix in range(self.width):
                    idx = iy*self.width+ix
                    z = (data.data[idx*2+1]*256+data.data[idx*2])/1000.0
                    if z!=0:
                        ## x, y are on the camera plane, z is the depth
                        #np_cloud[idx][0] = z*(ix-self.k[2])/self.k[0] #x
                        #np_cloud[idx][1] = z*(iy-self.k[5])/self.k[4] #y
                        #np_cloud[idx][2] = z
                        ## same coordinate as `/camera/depth/image_rect_raw`
                        ## y (left & right), z (up & down) are on the camera plane, x is the depth
                        np_cloud[idx][1] = -z*(ix-self.k[2])/self.k[0]
                        np_cloud[idx][2] = -z*(iy-self.k[5])/self.k[4]
                        np_cloud[idx][0] = z

            self.pcd.points = o3d.utility.Vector3dVector(np_cloud)
            ## publish as a ROS message
            # header = Header()
            # header.stamp = rospy.Time.now()
            # header.frame_id = "camera_depth_frame"
            # fields = self.FIELDS_XYZ

            # pc2_data = pc2.create_cloud(header, fields, np.asarray(np_cloud))
            # self.pub.publish(pc2_data)
            self.is_data_updated = True

    def cam_info_callback(self, data):
        if self.is_k_empty:
            for i in range(9):
                self.k[i] = data.K[i]

            if (self.k[0] != 0) and (self.k[4] != 0):
                self.is_k_empty = False



if __name__ == '__main__':
    rs = rs2o3d()

    rospy.init_node('rs2icp', anonymous=True)
    rospy.sleep(1)

    rospy.sleep(3)

    ## find ar_marker_90 first

    while rs.is_data_updated==False:
        rospy.spin()

    print(rs.width)
    print(rs.height)

    # o3d.io.write_point_cloud("./workspace.pcd", rs.pcd)

    # rate = rospy.Rate(10)
    o3d.visualization.draw_geometries([rs.pcd], 
                                      front = [ -0.65968065968571199, 0.75145078346514094, 0.011964416669851093 ],
                                      lookat = [ 3.5017595977975047, -2.1001878283924382, -0.13119255951620409 ],
                                      up = [ 0.029056782327167463, 0.0095939528158735781, 0.99953172009204305 ],
                                      zoom=0.14)