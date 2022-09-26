#!/usr/bin/env python3
import sys
sys.path.append('../../')
import copy

import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped
# tf has been deprecated
from tf.transformations import quaternion_from_matrix, quaternion_matrix

from utils.vision.rod_icp import rod_icp
from utils.vision.rs2o3d import rs2o3d
from utils.vision.rgb_camera import image_converter
from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose

class rod_info():
    def __init__(self):
        self.pose = Pose()
        self.l = 0.3
        self.r = 0.02
        self.box2d = np.zeros((4,2))

class rod_finder():
    def __init__(self, scene):
        # You need to initializing a node before instantiate the class
        self.scene = scene
        self.info = rod_info()
        self.t_rod_correction = np.array([[1, 0, 0, 0],\
                                          [0, 0, 1, 0],\
                                          [0,-1, 0, 0],\
                                          [0, 0, 0, 1]])

    def set_info(self, mat, l, r):
        q = quaternion_from_matrix(mat)
        o = mat[:3,3]
        self.info.pose.position.x = o[0]
        self.info.pose.position.y = o[1]
        self.info.pose.position.z = o[2]
        self.info.pose.orientation.x = q[0]
        self.info.pose.orientation.y = q[1]
        self.info.pose.orientation.z = q[2]
        self.info.pose.orientation.w = q[3]
        self.info.l = l
        self.info.r = r

    def add_to_scene(self):
        t_rod2world = pose2transformation(self.info.pose)
        t_rod_in_scene = np.dot(t_rod2world, np.linalg.inv(self.t_rod_correction))
        pose = transformation2pose(t_rod_in_scene)

        # updated = False
        cylinder_pose = PoseStamped()
        cylinder_pose.header.frame_id = "world"
        # assign cylinder's pose
        cylinder_pose.pose.position = copy.deepcopy(pose.position)
        cylinder_pose.pose.orientation = copy.deepcopy(pose.orientation)
        cylinder_name = "cylinder"
        # add_cylinder(self, name, pose, height, radius)
        self.scene.add_cylinder(cylinder_name, cylinder_pose, self.info.l, self.info.r)

        # ensuring collision updates are received
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # attached_objects = self.scene.get_attached_objects([cylinder_name])
            # print("attached_objects: ", end=',')
            # print(attached_objects)
            # is_attached = len(attached_objects.keys()) > 0

            is_known = cylinder_name in self.scene.get_known_object_names()

            # if (is_attached) and (is_known):
            #    return True
            if is_known:
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()
                
        return False

    def find_rod(self, ws_tf):
        ##-------------------##
        ## Detect the rod in the first place
        rs = rs2o3d()

        ri = rod_icp()
        ic = image_converter()

        ## There is depth data in the RS's buffer
        while rs.is_data_updated==False:
            print('waiting for depth data')
            rospy.sleep(0.1)

        print("depth_data_ready")

        ## transformation of the AR tag to world
        ## ar tag 90
        # t_ar2world = np.array([[0, 0, 1, 0],\
        #                        [1, 0, 0, 0],\
        #                        [0, 1, 0, 0.07],\
        #                        [0, 0, 0, 1]])

        ## ar tag 23
        # t_ar2world = np.array([[-0.02191529,  0.05454784,  0.99827063,  0.03081299],\
        #                        [ 0.99964885, -0.01368227,  0.02269318,  0.00288871],\
        #                        [ 0.01489647,  0.99841741, -0.05422884,  0.64434906],\
        #                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        t_ar2world = np.array([[-0.02191529,  0.05454784,  0.99827063,  0.01081299],\
                               [ 0.99964885, -0.01368227,  0.02269318,  0.00288871],\
                               [ 0.01489647,  0.99841741, -0.05422884,  0.64434906],\
                               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        # t_cam2ar = ws_tf.get_tf('ar_marker_90','front_cam_link')
        t_cam2ar = ws_tf.get_tf('ar_marker_23','front_cam_link')
        t_cam2world = np.dot(t_ar2world,t_cam2ar)
        ws_tf.set_tf("world", "front_cam_link", t_cam2world)


        ## There is RGB data in the RS's buffer (ic: image converter)
        while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

        print("rgb_data_ready")

        # h = ws_tf.get_tf('front_cam_depth_frame', 'ar_marker_90')
        h = ws_tf.get_tf('front_cam_depth_frame', 'ar_marker_23')
        ar_pos = h[:3,3]
        print("ar tag distance: %f"%ar_pos[0])
        print("ar pose is: {}".format(ar_pos))
        img = copy.deepcopy(ic.cv_image)
        
        ri.find_rod(rs.pcd, img, ar_pos, visualizing = True)
        self.info.box2d = ri.box2d
        
        ## broadcasting the rod's tf
        t_rod2cam = ri.rod_transformation

        t_rod_in_scene = np.dot(t_cam2world, t_rod2cam)

        # ## Overwrite rod's pose for test
        # t_rod_in_scene = np.array([[-0.98479743, -0.06750774,  0.16005225,  0.44197885],\
        #                            [-0.15012837, -0.13272416, -0.97971719, -0.08674831],\
        #                            [ 0.0873813,  -0.98885135,  0.12057159,  0.35553301],\
        #                            [ 0.,          0.,          0.,          1.        ]])


        t_rod2world = np.dot(t_rod_in_scene, self.t_rod_correction)

        ## apply correction matrix, because of the default cylinder orientation
        ws_tf.set_tf('world', 'rod', t_rod2world)

        ##-------------------##
        ## rod found, now you can save rod info
        self.set_info(t_rod2world, ri.rod_l, ri.rod_r)

        self.add_to_scene()
        ## Need time to initializing
        rospy.sleep(3)

    def test_with_files(path):
        import open3d as o3d
        import cv2

        img = cv2.imread("./"+ path +"/image.jpeg")
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pcd = o3d.io.read_point_cloud("./"+ path +"/workspace.pcd")
        # o3d.visualization.draw_geometries([pcd])
        ws_distance = 850/1000.0

        ri = rod_icp()
        ri.find_rod(pcd, img, ws_distance)
        # load
        ...

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_with_files(sys.argv[1])
    else:
        import moveit_commander
        from utils.workspace_tf   import workspace_tf, pose2transformation, transformation2pose

        rospy.init_node('wrap_wrap', anonymous=True)
        rospy.sleep(1)

        ws_tf = workspace_tf()
        
        moveit_commander.roscpp_initialize(sys.argv)
        scene = moveit_commander.PlanningSceneInterface()
        rod = rod_finder(scene)
        rod.find_rod(ws_tf)