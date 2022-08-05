## Use ICP to match a cylinder template to the captured point cloud data

import sys
sys.path.append('../../')
import copy

import numpy as np
from math import sin, cos, pi, sqrt
import matplotlib.pyplot as plt

import cv2
import open3d as o3d

from utils.vision.rgb_extract import object_mask

def draw_registration_result(source, target, transformation, additional_pcd = []):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp]+additional_pcd, 
    #                                   zoom=0.7, front=[-0.85, 0.5, 0.12], 
    #                                   lookat=[0.67,0.22,0], up=[0,0,1], left=1680)
    o3d.visualization.draw_geometries([source_temp, target_temp]+additional_pcd, 
                                      zoom=0.7, front=[-0.85, 0.5, 0.12], 
                                      lookat=[0.67,0.22,0], up=[0,0,1])

class rod_icp():
    def __init__(self,downsample_size = 0.005, eps=0.05, min_points=10, min_cluster_size = 500):
        self.rod_template = o3d.geometry.PointCloud()
        self.downsample_size = downsample_size
        ## DBSCAN parameters
        self.eps = eps
        self.min_points = min_points
        ## only pick those relatively larger cluster (including the rod)
        self.min_cluster_size = min_cluster_size
        self.rod_transformation = np.eye(4)
        self.rod_l = 0.30
        self.rod_r = 0.020
        self.box2d = []


    def create_cylinder_template(self, r=20/1000.0, l=200/1000.0):
        ## create a rod template here
        t_theta=60 # divide pi into 60 sampling points
        t_l = int(l*1000) # t_l sampling points along the rod
        obj_np_cloud = np.zeros((t_theta*t_l,3))
        for il in range(t_l):
            for it in range(t_theta):
                idx = il*t_theta+it

                obj_np_cloud[idx][1] = r*cos(it*pi/t_theta)
                obj_np_cloud[idx][0] = r*sin(it*pi/t_theta)
                obj_np_cloud[idx][2] = il/1000.0 - l/2

        self.rod_template.points = o3d.utility.Vector3dVector(obj_np_cloud)

    def find_corner(self, box, p, max_dist):
        ## give a point in projected pixel frame, find the closest point in 3D space
        ## use Manhattan distance in pixel frame
        for i in range(1,max_dist+1):
            for dy in range(-i, i+1):
                dx = i-abs(dy)
                ## Check if the 2D point is inside of the 2D mask
                x = int(p[0]+dx)
                y = int(p[1]+dy)
                if cv2.pointPolygonTest(box, (x,y), False) >= 0:
                    ## Check if the 3D point is inside of the labeled area
                    idx = y * self.width + x
                    pt_3d = self.raw_array[idx]
                    if (pt_3d[0] > self.om.x_min) and (pt_3d[0] < self.om.x_max) and \
                       (pt_3d[1] > self.om.y_min) and (pt_3d[1] < self.om.y_max) and \
                       (pt_3d[2] > self.om.z_min) and (pt_3d[2] < self.om.z_max):

                       return pt_3d

                ## check the other direction
                dx = -(i-abs(dy))
                ## Check if the 2D point is inside of the 2D mask
                x = int(p[0]+dx)
                y = int(p[1]+dy)
                if cv2.pointPolygonTest(box, (x,y), False) >= 0:
                    ## Check if the 3D point is inside of the labeled area
                    idx = y * self.width + x
                    pt_3d = self.raw_array[idx]
                    if (pt_3d[0] > self.om.x_min) and (pt_3d[0] < self.om.x_max) and \
                       (pt_3d[1] > self.om.y_min) and (pt_3d[1] < self.om.y_max) and \
                       (pt_3d[2] > self.om.z_min) and (pt_3d[2] < self.om.z_max):

                       return pt_3d

        ## nothing found
        return (-1, -1, -1)

    def dist_3d(self, p1, p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

    def find_rod(self, raw_pcd, img, ar_pos, visualizing = False):
        ## pcd is the raw point cloud data
        ## 1. based on the workspace distacne, 
        ## remove point cloud outside that distance
        ## ws_distance, measured in meters

        print("Finding the rod...")
        colorized = True

        ## ================
        ## 1. Remove points that beyond the robot
        self.raw_array = np.asarray(raw_pcd.points)
        ## om: object mask
        self.om = object_mask(self.raw_array, img, mask_color=(255,255,255))
        # print(raw_array.shape)

        if visualizing:
            color = np.resize(img, (img.shape[0]*img.shape[1],3))/255.0
            raw_pcd.colors = o3d.utility.Vector3dVector(color)

        ws_array = []
        ws_color = []
        for i in range(self.raw_array.shape[0]):
            ## x(ar_pos[0]) is the depth direction in RealSense coordiante
            ##
            if (self.raw_array[i][0] < ar_pos[0]) and (self.raw_array[i][2] > ar_pos[2]):
                ws_array.append([self.raw_array[i][0], self.raw_array[i][1], self.raw_array[i][2]])
                if visualizing:
                    ws_color.append([color[i][2], color[i][1], color[i][0]])

        ws_pcd = o3d.geometry.PointCloud()
        ws_pcd.points = o3d.utility.Vector3dVector(np.asarray(ws_array))
        if visualizing:
            ws_pcd.colors = o3d.utility.Vector3dVector(np.array(ws_color, dtype=np.float64))

        ## ================
        ## 2. Downsample pcd for clustering to reduce the computational load
        print('Downsample...')
        ds_pcd = ws_pcd.voxel_down_sample(voxel_size=self.downsample_size)

        ## ================
        ## 3. Apply DBSCAN clustering
        print('DBSCAN...')
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(ds_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=False))

        max_label = labels.max()

        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # print(colors.shape)
        # ds_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([ds_pcd], 
        #                               zoom=0.7, front=[-0.85, 0.5, 0.12], 
        #                               lookat=[0.67,0.22,0], up=[0,0,1])

        ## ===============
        ## 4. Select only the front most cluster as the rod/object
        ##    first value: number of points assigned to the label
        ##    second value: x distance (depth) associated with this label
        dist_sum = np.zeros(((max_label+1), 2))
        
        for idx, val in enumerate(labels):
            dist_sum[val,0] += 1
            dist_sum[val,1] += ds_pcd.points[idx][0]

        selected = 0
        min_dist = 10.0e6
        for i in range(max_label+1):
            ## must include more than 500 points (to avoid rope fragments)
            if (dist_sum[i,0] > self.min_cluster_size):
                dist = dist_sum[i,1]/dist_sum[i,0]
                # print(dist)
                if dist < min_dist:
                    selected = i
                    min_dist = dist

        print("select label", selected, ", with distance ", min_dist)
        selected_pcd = o3d.geometry.PointCloud()
        selected_array = np.zeros((int(dist_sum[selected,0]), 3))
        cnt = 0
        for idx, val in enumerate(labels):
            if val == selected:
                selected_array[cnt,0] = ds_pcd.points[idx][0]
                selected_array[cnt,1] = ds_pcd.points[idx][1]
                selected_array[cnt,2] = ds_pcd.points[idx][2]
                cnt += 1

                # creating a 3D bounding box for the rod
                if ds_pcd.points[idx][0] < self.om.x_min:
                    self.om.x_min = ds_pcd.points[idx][0]
                if ds_pcd.points[idx][0] > self.om.x_max:
                    self.om.x_max = ds_pcd.points[idx][0]
                if ds_pcd.points[idx][1] < self.om.y_min:
                    self.om.y_min = ds_pcd.points[idx][1]
                if ds_pcd.points[idx][1] > self.om.y_max:
                    self.om.y_max = ds_pcd.points[idx][1]
                if ds_pcd.points[idx][2] < self.om.z_min:
                    self.om.z_min = ds_pcd.points[idx][2]
                if ds_pcd.points[idx][2] > self.om.z_max:
                    self.om.z_max = ds_pcd.points[idx][2]

        selected_pcd.points = o3d.utility.Vector3dVector(np.asarray(selected_array))

        ## ================
        ## 5. Get the geometric information of the cluster

        ## apply point cloud bounding box
        self.om.apply_pc_mask()
        ## (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
        rect = self.om.extract_rod()
        self.width = img.shape[1]
        self.height = img.shape[0]

        ## estimate the dimension of the rod
        ## get four corner points of the box
        self.box2d = np.int0(cv2.boxPoints(rect))
        p = []
        for i in self.box2d:
            p.append(self.find_corner(self.box2d, i, 10))
        # p = self.find_corner(box, box[0], 10)
        l1 = self.dist_3d(p[0], p[1])
        l2 = self.dist_3d(p[1], p[2])

        ## d is the length of the rod
        ## r is the radius of the rod
        d = 0
        r = 0
        if l1 > l2:
            d = l1
            r = l2/2
        else:
            d = l2
            r = l1/2

        ## estimate the center of the rod
        cluster_center = (p[0]+p[1]+p[2]+p[3])/4.0

        ## ===============
        ## 6. Create a cylinder template for ICP
        self.create_cylinder_template(r=r, l=d)

        # o3d.visualization.draw_geometries([ds_pcd, self.rod_template], 
        #                               zoom=0.7, front=[-0.85, 0.5, 0.12], 
        #                               lookat=[0.67,0.22,0], up=[0,0,1])

        ## ===============
        ## 7. Apply ICP to register the rod's pose
        target = selected_pcd
        source = self.rod_template
        threshold = 0.02
        trans_init = np.asarray(
                    [[-1, 0, 0, cluster_center[0]],
                     [ 0, 0, 1, cluster_center[1]],
                     [ 0, 1, 0, cluster_center[2]],
                     [ 0, 0, 0, 1]])

        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        self.rod_transformation = reg_p2p.transformation
        print(reg_p2p.transformation)
        self.rod_l = d
        self.rod_r = r

        cv2.imwrite('image.jpg', img)

        if visualizing == True:
            print("")
            # # draw_registration_result(source, raw_pcd, reg_p2p.transformation)
            # # draw_registration_result(source, target, reg_p2p.transformation)
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=cluster_center)
            draw_registration_result(source, ws_pcd, reg_p2p.transformation, [axis_pcd])
