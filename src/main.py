import sys
import copy

import numpy as np
from math import pi,sin,cos,asin,acos, degrees

from utils.vision.rod_finder import rod_finder

import open3d as o3d
import cv2

## run `roslaunch rs2pcl ar_bc_test.launch` first
## the default function
def main():
    import rospy
    from utils.vision.rs2o3d import rs2o3d
    # from utils.vision.workspace_tf import workspace_tf
    from utils.vision.rgb_camera import image_converter

    import moveit_commander
    from utils.robot.rod_info        import rod_info
    # from utility.detect_cable    import cable_detection
    from utils.robot.workspace_ctrl  import move_yumi
    from utils.robot.jointspace_ctrl import joint_ctrl
    from utils.robot.path_generator  import path_generator, step_back
    from utils.robot.gripper_ctrl    import gripper_ctrl
    from utils.robot.visualization   import marker

    from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose

    from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
    from transforms3d import euler
    import moveit_msgs.msg

    import tf
    from tf.transformations import quaternion_from_matrix, quaternion_matrix

    bc = tf.TransformBroadcaster()

    rospy.init_node('wrap_wrap', anonymous=True)
    rate = rospy.Rate(10)
    rospy.sleep(1)

    pg = path_generator()
    gripper = gripper_ctrl()
    goal = marker()

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

    ##-------------------##
    ## reset the robot
    gripper.l_open()
    gripper.r_open()
    j_ctrl.robot_default_l_low()
    j_ctrl.robot_default_r_low()

    # rospy.sleep(5)
    # h5_6 = ws_tf.get_tf('yumi_link_5_l','yumi_link_6_l')
    # h6_7 = ws_tf.get_tf('yumi_link_6_l','yumi_link_7_l')
    # # print(h5_6)
    # print(h6_7)
    # # theta_5 = degrees(asin(-h5_6[0,1]))
    # # alpha_5 = degrees(asin(-h5_6[1,2]))
    # # theta_6 = degrees(asin(-h6_7[0,1]))
    # # alpha_6 = degrees(asin(-h6_7[1,2]))
    # # print('theta_5: {}, alpha_5: {}, theta_6: {}, alpha_6: {}'.format(theta_5, alpha_5, theta_6, alpha_6))

    gripper.l_open()
    gripper.r_open()

    rospy.sleep(3)

    # print(ctrl_group[0].get_current_joint_values())
    # print(robot.get_current_state())

    ##-------------------##
    ## Detect the rod in the first place
    rs = rs2o3d()

    rf = rod_finder()
    ic = image_converter()

    ## There is depth data in the RS's buffer
    while rs.is_data_updated==False:
        rate.sleep()

    print("depth_data_ready")


    ## transformation of the AR tag to world
    t_ar2world = np.array([[0, 0, 1, 0],\
                           [1, 0, 0, 0],\
                           [0, 1, 0, 0.07],\
                           [0, 0, 0, 1]])
    t_cam2ar = ws_tf.get_tf('ar_marker_90','camera_link')
    t_cam2world = np.dot(t_ar2world,t_cam2ar)
    ws_tf.set_tf("world", "camera_link", t_cam2world)


    ## There is RGB data in the RS's buffer (ic: image converter)
    while ic.has_data==False:
        rate.sleep()

    print("rgb_data_ready")

    h = ws_tf.get_tf('camera_depth_frame', 'ar_marker_90')
    ws_distance = h[0,3]
    print(ws_distance)
    img = copy.deepcopy(ic.cv_image)
    
    # rf.find_rod(rs.pcd, img, ws_distance)
    rf.rod_transformation = np.array([[ 0.9935388,   0.10384128, -0.04579992,  0.60588046],\
                                      [ 0.02949805,  0.15340868,  0.98772245,  0.00633545],\
                                      [ 0.10959247, -0.98269159,  0.14935436,  0.04958995],\
                                      [ 0.,          0.,          0.,          1.,       ]])
    rf.rod_l = 0.291540804700394
    rf.rod_r = 0.02086036592098056

    t_rod_correction = np.array([[-1, 0, 0, 0],\
                                 [0, 0, -1, 0],\
                                 [0, -1, 0, 0],\
                                 [0, 0, 0, 1]])
    ## broadcasting the rod's tf
    t_rod2cam = rf.rod_transformation

    t_rod_in_scene = np.dot(t_cam2world, t_rod2cam)
    t_rod2world = np.dot(t_rod_in_scene, t_rod_correction)

    ## apply correction matrix, because of the default cylinder orientation
    ws_tf.set_tf('world', 'rod', t_rod2world)

    ##-------------------##
    ## rod found, start to do the first wrap

    rod = rod_info(scene, rate)
    rod.set_info(t_rod_in_scene, rf.rod_l, rf.rod_r)

    rod.scene_add_rod()
    ## Need time to initializing
    rospy.sleep(3)

    # # key = input("Help me to put the cable on the rod! (q to quit)")
    # # if key =='q':
    # #     return

    ##-------------------##
    ## generate spiral here
    step_size = 0.02
    r = rod.rod_state.r
    ## l is the parameter to be tuned
    l = 2*pi*r + 0.1
    curve_path = pg.generate_nusadua(t_rod2world, l, r, step_size)

    pg.publish_waypoints(curve_path)

    j_start_value = 2*pi-2.5
    yumi.pose_with_restrict(0, curve_path[0], j_start_value)
    rospy.sleep(2)

    for i in range(len(curve_path)-1):
        angle = j_start_value - 2*pi/len(curve_path)*(i+1)
        yumi.pose_with_restrict(0, curve_path[i+1], angle)
        # rospy.sleep(0.1)


    # goal.show(starting_pose)
    # yumi.go_to_pose_goal(ctrl_group[0], starting_pose)
    # print(ctrl_group[0].get_current_joint_values())

    # ## from default position move to the rope starting point
    # stop  = curve_path[0]
    # start = [stop[0], stop[1] + 0.25, stop[2]]
    # cartesian_plan, fraction = yumi.plan_cartesian_traj(ctrl_group, 0, [start, stop])
    # yumi.execute_plan(cartesian_plan, ctrl_group[0])
    # print("go to pose have the cable in between gripper: ", end="")
    # rospy.sleep(2)

    # ## grabbing the rope
    # gripper.l_close()

    # ## wrapping
    ...

    # gripper.l_open()
    # gripper.r_open()

    # start = curve_path[-1]
    # stop = [start[0], start[1], 0.1]

    # path = [start, stop]
    # cartesian_plan, fraction = yumi.plan_cartesian_traj(ctrl_group, 0, path)
    # yumi.execute_plan(cartesian_plan, ctrl_group[0])

    # # gripper.l_open()
    # # gripper.r_open()


def test_with_files(path):
    img = cv2.imread("./"+ path +"/image.jpeg")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pcd = o3d.io.read_point_cloud("./"+ path +"/workspace.pcd")
    # o3d.visualization.draw_geometries([pcd])
    ws_distance = 850/1000.0

    rf = rod_finder()
    rf.find_rod(pcd, img, ws_distance)
    # load
    ...

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_with_files(sys.argv[1])
    else:
        main()