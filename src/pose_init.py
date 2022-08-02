import sys
import copy

import numpy as np

from utils.vision.rod_icp import rod_icp

def main(rod, ws_tf):
    from utils.vision.rs2o3d import rs2o3d
    from utils.vision.rgb_camera import image_converter

    ##-------------------##
    ## Detect the rod in the first place
    rs = rs2o3d()

    ri = rod_icp()
    ic = image_converter()

    ## There is depth data in the RS's buffer
    while rs.is_data_updated==False:
        print('waiting for depth data')
        rate.sleep()

    print("depth_data_ready")

    ## transformation of the AR tag to world
    t_ar2world = np.array([[0, 0, 1, 0],\
                           [1, 0, 0, 0],\
                           [0, 1, 0, 0.07],\
                           [0, 0, 0, 1]])
    t_cam2ar = ws_tf.get_tf('ar_marker_90','front_cam_link')
    t_cam2world = np.dot(t_ar2world,t_cam2ar)
    ws_tf.set_tf("world", "front_cam_link", t_cam2world)


    ## There is RGB data in the RS's buffer (ic: image converter)
    while ic.has_data==False:
        print('waiting for RGB data')
        rate.sleep()

    print("rgb_data_ready")

    h = ws_tf.get_tf('front_cam_depth_frame', 'ar_marker_90')
    ar_pos = h[:3,3]
    print("ar tag distance: %f"%ar_pos[0])
    img = copy.deepcopy(ic.cv_image)
    
    ri.find_rod(rs.pcd, img, ar_pos, visualizing = False)

    t_rod_correction = np.array([[1, 0, 0, 0],\
                                 [0, 0, 1, 0],\
                                 [0,-1, 0, 0],\
                                 [0, 0, 0, 1]])
    ## broadcasting the rod's tf
    t_rod2cam = ri.rod_transformation

    t_rod_in_scene = np.dot(t_cam2world, t_rod2cam)

    # ## Overwrite rod's pose for test
    # t_rod_in_scene = np.array([[-0.98479743, -0.06750774,  0.16005225,  0.44197885],\
    #                            [-0.15012837, -0.13272416, -0.97971719, -0.08674831],\
    #                            [ 0.0873813,  -0.98885135,  0.12057159,  0.35553301],\
    #                            [ 0.,          0.,          0.,          1.        ]])


    t_rod2world = np.dot(t_rod_in_scene, t_rod_correction)

    ## apply correction matrix, because of the default cylinder orientation
    ws_tf.set_tf('world', 'rod', t_rod2world)

    ##-------------------##
    ## rod found, now you can save rod info
    rod.set_info(t_rod_in_scene, ri.rod_l, ri.rod_r)

    rod.scene_add_rod()
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
        import rospy
        import moveit_commander
        from utils.robot.rod_info import rod_info
        from utils.workspace_tf   import workspace_tf, pose2transformation, transformation2pose

        rospy.init_node('wrap_wrap', anonymous=True)
        rate = rospy.Rate(10)
        rospy.sleep(1)

        ws_tf = workspace_tf(rate)
        
        moveit_commander.roscpp_initialize(sys.argv)
        scene = moveit_commander.PlanningSceneInterface()
        rod = rod_info(scene, rate)
        main(rod, ws_tf)