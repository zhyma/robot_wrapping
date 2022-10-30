# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
import rospy
import numpy as np
from math import sin, cos, pi

import time

from math import pi

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

import open3d as o3d

from cv_bridge import CvBridge, CvBridgeError
import cv2

#from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromOpen3dToRos

bridge = CvBridge()

class rs2pc():
    def __init__(self):
        self.is_k_empty = True
        self.is_data_updated = False
        self.k = [0]*9 # camera's intrinsic parameters
        self.cam_sub = rospy.Subscriber("/front_cam/depth/camera_info", CameraInfo, self.cam_info_callback)
        # self.img_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.img_callback)
        self.img_sub = rospy.Subscriber("/front_cam/aligned_depth_to_color/image_raw", Image, self.img_callback)
        self.pcd = o3d.geometry.PointCloud()

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

def image_callback(msg):
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError:
        # print(e)
        ...
    else:
        cv2.imwrite("image.jpeg", cv2_img)



def main():
    rs = rs2pc()

    rospy.init_node('rs2icp', anonymous=True)
    rospy.sleep(1)

    rospy.Subscriber("/front_cam/color/image_raw", Image, image_callback)

    rospy.sleep(3)
    while rs.is_data_updated==False:
        rospy.spin()

    o3d.io.write_point_cloud("./workspace.pcd", rs.pcd)


if __name__ == '__main__':
    
    main()