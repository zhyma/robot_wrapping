## run `roslaunch rs2pcl demo.launch` first

import sys, copy, pickle, os, rospy, moveit_commander

import numpy as np
from math import pi

from utils.workspace_tf          import workspace_tf, pose2transformation, transformation2pose

from utils.robot.rod_finder      import rod_finder
from utils.robot.workspace_ctrl  import move_yumi
from utils.robot.jointspace_ctrl import joint_ctrl
from utils.robot.path_generator  import path_generator
from utils.robot.gripper_ctrl    import gripper_ctrl
from utils.robot.interpolation   import interpolation
from utils.robot.visualization   import marker

from utils.vision.rgb_camera     import image_converter
from utils.vision.rope_detect    import rope_detect
from utils.vision.adv_check      import check_adv
from utils.vision.len_check      import check_len

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

def tf_with_offset(tf, offset):
    rot = tf[:3, :3]
    new_tf = copy.deepcopy(tf)
    d_trans = np.dot(rot, np.array(offset))
    for i in range(3):
        new_tf[i, 3]+=d_trans[i]

    return new_tf

def pose_with_offset(pose, offset):
    ## the offset is given based on the current pose's translation
    tf = pose2transformation(pose)
    new_tf = tf_with_offset(tf, offset)
    new_pose = transformation2pose(new_tf)

    ## make sure it won't go too low
    if new_pose.position.z < 0.03:
            new_pose.position.z = 0.03
    return new_pose

class robot_winding():
    def __init__(self):
        self.gripper = gripper_ctrl()
        ##-------------------##
        ## initializing the moveit 
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()
        self.ws_tf = workspace_tf()

        self.ctrl_group = []
        self.ctrl_group.append(moveit_commander.MoveGroupCommander('left_arm'))
        self.ctrl_group.append(moveit_commander.MoveGroupCommander('right_arm'))
        self.j_ctrl = joint_ctrl(self.ctrl_group)
        ## initialzing the yumi motion planner
        self.yumi = move_yumi(self.robot, self.scene, self.ctrl_group, self.j_ctrl)
        self.pg = path_generator()

        self.marker = marker()

        self.ic = image_converter()
        while self.ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

        ## control parameters
        self.adv = -1.0
        self.r   = -1.0
        self.len = -1.0

        if os.path.exists('param.txt'):
            ## file format: '0.1,0.2,0.3'
            ## values are advance, r, and L' (L=2*\pi*R+L')
            with open('param.txt', 'r') as file:
                line = file.readline()
                self.adv = float(line.split(',')[0])
                self.r   = float(line.split(',')[1])
                self.len = float(line.split(',')[2])

        else:
            ## file does not exist, create a new one
            with open('param.txt', 'w') as file:
                file.write(str(self.adv)+','+str(self.r)+','+str(self.len))

        ## initializing the last feedback state and the last frontier
        self.last_adv_fb   = -10
        self.last_len_fb   = -10
        self.last_frontier = None

    def move2pt(self, point, j6_value, group = 0):
        q = self.yumi.ik_with_restrict(group, point, j6_value)
        if type(q) is int:
            print('No IK found')
            return -1
        else:
            self.j_ctrl.robot_setjoint(group, q)

    def pts2qs(self, pts, j_start_value, d_j):
        ## solve each workspace point to joint values
        q_knots = []
        n_pts = len(pts)
        last_j_angle = j_start_value + d_j*(n_pts-1)
        solved_pts = 0
        no_ik_found = []
        for i in range(n_pts-1, -1, -1):
            # print('point {} is\n{}\n{}'.format(i, pts[i], last_j_angle))
            q = self.yumi.ik_with_restrict(0, pts[i], last_j_angle)
            ## reversed searching, therefore use -d_j
            last_j_angle -= d_j
            if q==-1:
                ## no IK solution found, remove point
                no_ik_found.append(i)
            else:
                # print("point {} solved".format(i))
                q_knots.insert(0, q)
                solved_pts += 1

        if len(no_ik_found) > 0:
            print("No IK solution is found at point: ",end='')
            for i in no_ik_found:
                print(str(i), end=',')
            print("\n")
        if solved_pts < n_pts/2:
            return -1

        return q_knots

    def rope_holding(self, z):
        ## go to the rope's fixed end

        ## by given the expected z=0.12 value
        ## find the grasping point along y-axis
        pos = self.rope.y_estimation(self.ic.cv_image, z, end=1)

        stop = transformation2pose(np.array([[ 1,  0, 0, pos[0]],\
                                             [ 0,  0, 1, pos[1]],\
                                             [ 0, -1, 0, pos[2]],\
                                             [ 0,  0, 0, 1    ]]))
        ## -0.04 is the offset for better capturing (opening)
        ## -0.14 is the offset for fingers (finger tip to finger belly)
        stop = pose_with_offset(stop, [-0.04, 0, -0.14])
        start = pose_with_offset(stop, [0, 0, -0.03])

        ## group=1, right
        # self.move_p2p(start, stop, -0.5, group=1)
        self.move2pt(start, -0.5, group=1)
        rospy.sleep(2)
        self.move2pt(stop, -0.5, group=1)
        rospy.sleep(2)

        self.gripper.r_close()
        rospy.sleep(2)

        hold = pose_with_offset(stop, [-0.03, 0, -0.03])

        print("move out of the other finger's workspace")
        self.move2pt(hold, -0.5, group=1)
        rospy.sleep(2)

    def winding(self, pretuned_demo=False, new_learning=False, learning=False, execute=True):
        ##---- winding task entrance here ----##
        ## reset -> load info -> wrapping step(0) -> evaluate -> repeate wrapping to evaluate 3 times -> back to starting pose
        
        if execute:
            self.reset()

        ## recover rod's information from the saved data
        rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            rod.info = pickle.load(handle)
            # print('load rod:{}'.format(rod.info.pose))

        self.rope = rope_detect(rod.info)
        with open('rope_info.pickle', 'rb') as handle:
            self.rope.info = pickle.load(handle)
            # print('load rope: with color: {}, diameter: {}'.format(self.rope.info.hue, self.rope.info.diameter))

        rod.add_to_scene()
        # print('rod added to scene')

        ## get transform rod2world here
        t_rod2world = pose2transformation(rod.info.pose)
        self.ws_tf.set_tf('world', 'rod', t_rod2world)

        ##-------------------##
        ## generate spiral here, two parameters to tune: advance and l = 2*pi*r+l'
        if (self.adv < -0.05 or self.r < 0 or self.len < 0) or new_learning:
            ## No previous parameters or start a fresh learning, setup default parameters
            self.adv = 0.02 ## meter
            self.r   = rod.info.r * 1.5
            self.len = 0.06
            with open('param.txt', 'w') as file:
                file.write(str("{:.3f}".format(self.adv))+','+str("{:.3f}".format(self.r))+','+str("{:.3f}".format(self.len)))

        if pretuned_demo:
            ## demo wrapping, use constants (l=0.18)
            self.adv = 0.0
            self.r   = 0.02
            self.len = 0.0544
            
        print("rod's radius is:{}".format(rod.info.r))
        # # l = 2*pi*r + 0.10
        # # print('Estimated L is:{}'.format(l))
        # l=0.18

        print("====starting the first wrap")
        # wrapping_pose = self.rope.find_frontier(self.ic.cv_image, t_rod2world)
        # t_wrapping = pose2transformation(wrapping_pose)
        # wrapping_pose.position.z = 0.18
        # self.marker.show(wrapping_pose)

        # Left and right arm are not mirrored. The left hand is having some problem
        # with reaching points that are too low. Give it a 
        if execute:
            self.rope_holding(0.12)

        ## let's do a few rounds
        i = 0
        with open("log.txt", 'a') as file:
            file.write(",,,new wraps\n")
        while i < 3:
            last_adv = self.adv
            last_r   = self.r
            last_len = self.len
            print("\n========")
            print("Current parameters are:\nadv: {:.3}\nr: {:.3}\nlen: {:.3}".format(last_adv, last_r, last_len))
            with open("log.txt", 'a') as file:
                file.write("{:.3},{:.3},{:.3},".format(last_adv, last_r, last_len))

            param_updated = False
            ## find the left most wrap on the rod
            wrapping_pose = self.rope.find_frontier(self.ic.cv_image, t_rod2world)
            t_wrapping = pose2transformation(wrapping_pose)
            wrapping_pose.position.z = 0.25
            self.marker.show(wrapping_pose)

            ## does not distinguish demo or learning here
            ## for demo, l = 0.18
            l = self.r*pi*2 + self.len
            result = self.step(t_wrapping, self.r, l, self.adv, debug = True, execute=execute)

            if pretuned_demo:
                i += 1
            elif (new_learning or learning):
                print("***Learning phrase...***")
                ## for new_learning==True (start a new learning)
                if result < 0:
                    ## NO IK found, need to tune self.len
                    if self.len > 0.02: ## should always be roughly larger than the size of the gripper
                        self.len -= 0.01
                        print("Next self.len to test is {}".format(self.len))
                        param_updated = True
                        with open("log.txt", 'a') as file:
                            file.write("No IK found. Reduce L'.,")
                    else:
                        print('Safety distance between the gripper and the rod cannot be guaranteed!')
                        self.r -= 0.005 ## try to reduce the r instead?
                        self.len = 0.06
                        with open("log.txt", 'a') as file:
                            file.write("No IK found. Safety distance reached. Reduce r.,")

                else:
                    with open("log.txt", 'a') as file:
                            file.write("Done wrap No. {}.,".format(i))
                    
                    print("***Do one wrap successfully***")

                    len_fb = check_len(self.ic.cv_image, rod.info.box2d, self.rope.info.hue, self.rope.info.diameter)
                    print("Extra length is: {}".format(len_fb))

                    ## update L'
                    if self.last_len_fb > 0:
                        if ((abs(len_fb-self.last_len_fb)/len_fb > 0.1) or len_fb > 20) and (len_fb-20 > 0):
                            ## len_new = len - k2*(len_feedback - threshold2)
                            self.r = self.r - 0.001*(len_fb-20)
                            self.len = 0.06 ## having a new self.r, then start to search L' from beginning
                            print("Next self.r to test is {}".format(self.r))
                            with open("log.txt", 'a') as file:
                                file.write("change r to {:.3},".format(self.r))
                            param_updated = True
                        else:
                            print('The selection of r becomes stable')
                            with open("log.txt", 'a') as file:
                                file.write("r becomes stable,")
                    else:
                        self.last_len_fb = len_fb

                    ## update adv starting from the second wrap
                    if i > 0:
                        ## skip the first wrap (for adv)?
                        ## get feedback
                        adv_fb = check_adv(self.ic.cv_image, rod.info.box2d, self.rope.info.hue, self.rope.info.diameter)
                        
                        print("Tested advnace is: {}, ".format(adv_fb))
                        
                        if self.last_adv_fb > 0:
                            if (abs(adv_fb-self.last_adv_fb)/adv_fb > 0.1) or adv_fb > 0.2:
                                ## adv_new = adv - k1*(adv_feedback - threshold1)
                                self.adv = self.adv - 0.01*(adv_fb - 0.2)
                                print("Next self.adv to test is {}".format(self.adv))
                                param_updated = True
                                with open("log.txt", 'a') as file:
                                    file.write("change adv to {:.3},".format(self.adv))
                            else:
                                print('The selection of advance becomes stable')
                        else:
                            self.last_adv_fb = adv_fb

                    i += 1

                if param_updated:
                    with open('param.txt', 'w') as file:
                        file.write(str("{:.3f}".format(last_adv))+','+str("{:.3f}".format(last_r))+','+str("{:.3f}".format(last_len)))
        
            if (not pretuned_demo) and (new_learning or learning):
                ## one wrap/trial is done
                with open("log.txt", 'a') as file:
                    file.write("\n")

        if execute:
            self.reset()

    def step(self, center_t, r, l, advance, debug = False, execute=True):
        curve_path = self.pg.generate_nusadua(center_t, r, l, advance)

        self.pg.publish_waypoints(curve_path)

        finger_offset = [0, 0, -0.10]


        ## the arbitary value (but cannot be too arbitary) of the starting value of the last/wrist joint
        j_start_value = 2*pi-2.5
        j_stop_value = j_start_value - 2*pi

        ## preparing the joint space trajectory
        q1_knots = []
        last_j_angle = 0.0## wrapping

        # print('IK for spiral')

        ## do the ik from the last point, remove those with no solution.
        n_pts = len(curve_path)
        print("planned curve_path: {}".format(n_pts))

        d_j = -2*pi/n_pts
        # print("d_j is {}".format(d_j))
        # skip the first waypoint (theta=0) and the last one (theta=2\pi)
        q1_knots = self.pts2qs(curve_path[1:-1], j_start_value+d_j, d_j)
        if type(q1_knots) is int:
            print('not enough waypoints for the spiral, skip')
            return -1
        print("sovled q1:{}".format(len(q1_knots)))

        print("solved curve_path: {}".format(len(curve_path)))
        ## solution found, now execute
        n_samples = 10
        dt = 2
        j_traj_1 = interpolation(q1_knots, n_samples, dt)

        poses = [Pose()]*7
        ## poses[0]: entering point
        ## poses[1]: grabbing point
        ## poses[2]: pushback position (away from the rope, x and y directions)
        ## poses[3]: pushback position (away from the rope, x direction)
        ## poses[4]: pushback position (away from the rope, x direction. adjust according to current rope's pose)
        ## poses[5]: pushback position (right under the rod)
        gp_pos = self.rope.gp_estimation(self.ic.cv_image, end=0, l=l-0.02)

        ## only need the orientation of the gripper
        ## start with generating poses[1]
        poses[1] = copy.deepcopy(curve_path[0])
        poses[1].position.x = gp_pos[0]
        poses[1].position.y = gp_pos[1]
        poses[1].position.z = gp_pos[2]
        ## z is the offset along z-axis
        poses[1] = pose_with_offset(poses[1], finger_offset)

        # self.marker.show(curve_path[0])
        ## based on the frame of link_7 (not the frame of the rod)
        ## z pointing toward right
        poses[0] = pose_with_offset(poses[1], [-0.01, 0, -0.06])

        ## for straightening the rope
        pose = pose_with_offset(curve_path[-1], [0, 0.10, 0])
        line_path = self.pg.generate_line(curve_path[-2], pose)
        self.pg.publish_waypoints(curve_path + line_path)
        q2_knots = self.pts2qs(line_path, j_stop_value, 0)
        if type(q2_knots) is int:
            print('not enough waypoints for straighten the rope, skip')
            return -2
        j_traj_2 = interpolation(q2_knots, 2, 0.2)

        gp_pos = self.rope.gp_estimation(self.ic.cv_image, end=0, l=l)
        poses[2] = transformation2pose(np.array([[ 0,  1, 0, poses[0].position.x+0.04],\
                                                 [ 0,  0,-1, poses[0].position.y],\
                                                 [-1, 0, 0, 0.12],\
                                                 [ 0,  0, 0, 1    ]]))

        poses[3] = pose_with_offset(poses[2], [0, 0, 0.06])
        poses[4] = copy.deepcopy(poses[3])
        poses[4].position.y = gp_pos[1] - finger_offset[2]
        poses[5] = copy.deepcopy(poses[4])
        poses[5].position.x = curve_path[0].position.x

        j6_values = [j_start_value]*2+[j_stop_value]+[j_stop_value + pi/2]*5+[j_start_value] 

        js_values = []
        pose_seq = [0, 1, 0, 2, 3, 4, 5, 2, 0]
        for i in range(len(pose_seq)):
            q = self.yumi.ik_with_restrict(0, poses[pose_seq[i]], j6_values[i])
            if type(q) is int:
                print('No IK found for facilitate steps')
                return -1
            else:
                js_values.append(q)


        # menu  = '=========================\n'
        # menu += '0. NO!\n'
        # menu += '1. execute\n'
        # choice = input(menu)
        # if choice == 1:
        #     execute = True
        # else:
        #     execute = False

        if execute:
            # print('move closer to the rope')
            self.j_ctrl.robot_setjoint(0, js_values[0])
            rospy.sleep(2)
            self.j_ctrl.robot_setjoint(0, js_values[1])
            ## grabbing the rope
            self.gripper.l_close()
            rospy.sleep(2)

            # ## test the first two points of the curve
            # self.j_ctrl.exec(0, [j_traj_1[0], j_traj_1[1]], 0.2)

            # print('wrapping...')
            self.j_ctrl.exec(0, j_traj_1, 0.2)

            # print('straighten out the rope')
            ## straighten out the rope
            # self.marker.show(pt_2)
            self.j_ctrl.exec(0, j_traj_2, 0.2)

            rospy.sleep(2)
            self.gripper.l_open()

            # print('move out from the grasping pose')
            ## left grippermove to the side
            self.j_ctrl.robot_setjoint(0, js_values[2])
            # rospy.sleep(2)

            # print('push the rope back a little bit')
            for i in [3,4,5,6,7]:
                rospy.sleep(2)
                self.j_ctrl.robot_setjoint(0, js_values[i])

            # print('move out of the view')
            ## left grippermove to the side
            self.j_ctrl.robot_setjoint(0, js_values[8])
            rospy.sleep(2)

            return 0 ## execute successfully (hopefully)

        return -1 ## for any other reason failed to execute

    def reset(self):
        ##-------------------##
        ## reset the robot
        print('reset the robot pose')
        self.gripper.l_open()
        self.gripper.r_open()
        self.j_ctrl.robot_default_l_low()
        self.j_ctrl.robot_default_r_low()

        self.gripper.l_open()
        self.gripper.r_open()

        rospy.sleep(3)
        print('reset done')


if __name__ == '__main__':
    run = True
    menu  = '=========================\n'
    menu += '1. Reset the robot\n'
    menu += '2. Wrapping motion demo with pretuned parameters\n'
    menu += '3. Wrapping motion demo with current parameters\n'
    menu += '4. Start a fresh new learning\n'
    menu += '5. Continue previous learning\n'
    menu += '0. Exit\n'
    menu += 'Your input:'

    rospy.init_node('wrap_wrap', anonymous=True)
    # rate = rospy.Rate(10)
    rospy.sleep(1)

    rw = robot_winding()
    while run:
        choice = input(menu)
        
        if choice in ['1', '2', '3', '4', '5']:
            if choice == '1':
                ## reset the robot
                rw.reset()
            elif choice == '2':
                ## show pretuned demo
                rw.winding(pretuned_demo=True,  new_learning=False, learning=False)
            elif choice == '3':
                ## wrap with current parameters
                rw.winding(pretuned_demo=False, new_learning=False, learning=False)
            elif choice == '4':
                ## start a new learning, tuning parameters automatically
                rw.winding(pretuned_demo=False, new_learning=True,  learning=True)
            elif choice == '5':
                ## continue previous learning, tuning parameters automatically
                rw.winding(pretuned_demo=False, new_learning=False, learning=True)

        else:
            ## exit
            run = False
