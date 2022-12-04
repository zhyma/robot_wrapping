## run `roslaunch rs2pcl demo.launch` first

import sys, copy, pickle, os, time, rospy, moveit_commander, cv2

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

class robot_wrapping():
    def __init__(self):
        self.options = ['demo_current', 'new_learning', 'continue_previous']
        self.time_str = time.strftime('%m-%d_%H-%M-%S',time.localtime(time.time()))
        self.wrap_no = 0 ## number of wrap

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

        self.wrap_success = True

        [self.adv_s, self.r_s, self.lp_s] = [-1.0, -1.0, -1.0] ## L prime, or L', 's' stand for stable result
        [self.adv,   self.r,   self.lp  ] = [-1.0, -1.0, -1.0]
        [self.adv_n, self.r_n, self.lp_n] = [-1.0, -1.0, -1.0] ## L prime, or L', 'n' stand for next to test
        [self.last_adv_fb, self.last_len_fb] = [-10, -10]

        ## recover rod's information from the saved data
        self.rod = rod_finder(self.scene)

        with open('rod_info.pickle', 'rb') as handle:
            self.rod.info = pickle.load(handle)
            # print('load rod:{}'.format(rod.info.pose))

        self.rope = rope_detect(self.rod.info)
        with open('rope_info.pickle', 'rb') as handle:
            self.rope.info = pickle.load(handle)
            # print('load rope: with color: {}, diameter: {}'.format(self.rope.info.hue, self.rope.info.diameter))

        self.rod.add_to_scene()
        # print('rod added to scene')

        ## get transform rod2world here
        self.t_rod2world = pose2transformation(self.rod.info.pose)
        self.ws_tf.set_tf('world', 'rod', self.t_rod2world)

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

    def wrapping(self, input_option, param_filename='', execute=True):
        ##---- wrapping task entrance here ----##
        ## reset -> load info -> wrapping step(0) -> evaluate -> repeate wrapping to evaluate 3 times -> back to starting pose
        if input_option == 'continue_previous' or input_option == 'demo_current':
            if len(param_filename) <= 0:
                ## use the current variables of the class
                if self.r < 0:
                    print("No previous learning?")
                    return
            else:
                ## load from a file
                if not os.path.exists('./save/'+param_filename):
                    print("No such a file.")
                    return
                with open('./save/'+param_filename, 'r') as file:
                    lines = file.read().split('\n')
                    [self.adv_s, self.r_s, self.lp_s]    = [float(i) for i in lines[0].split(',')]
                    [self.adv_n, self.r_n, self.lp_n]    = [float(i) for i in lines[1].split(',')]
                    [self.last_adv_fb, self.last_len_fb] = [float(i) for i in lines[2].split(',')]
        elif input_option == 'new_learning':
             ## start a new one with default parameters.
            ## three parameters to tune: advance, r and L'. L = 2*pi*r+L'
            [self.adv_n, self.r_n, self.lp_n] = [0.02, self.rod.info.r * 1.5, 0.06] ## meter
            [self.last_adv_fb, self.last_len_fb] = [-10, -10]
        else:
            print("No such an option.")
            return

        print("====starting a wrap")
        
        ## let's do a few rounds
        with open("./save/log.txt", 'a') as file:
            file.write("====new wraps: "+time_str+" @ "+str(self.wrap_no)+"====\n")

        self.wrap_success = False

        param_stable = [False, False]
        print("\n========")

        param_updated = False
        ## find the left most wrap on the rod
        wrapping_pose = self.rope.find_frontier(self.ic.cv_image, self.t_rod2world)
        t_wrapping = pose2transformation(wrapping_pose)
        wrapping_pose.position.z = 0.25
        self.marker.show(wrapping_pose)

        use_last_img = False
        if self.wrap_success==False:
            use_last_img = True

        while self.wrap_success == False:
            [self.adv, self.r, self.lp] = [self.adv_n, self.r_n, self.lp_n]
            l = self.r*pi*2 + self.lp
            print("Current wrap {}, @{}\nparameters are:\nadv: {:.1f}\nr: {:.1f}\nlen: {:.1f}".format(self.wrap_no, self.time_str, self.adv*1000, self.r*1000, self.lp*1000))
            with open("./save/log.txt", 'a') as file:
                file.write("adv: {:.1f}, r: {:.1f}, lp: {:.1f},".format(self.adv*1000, self.r*1000, self.lp*1000))
            result = self.step(self.wrap_no, t_wrapping, self.r, l, self.adv, debug = True, execute=execute, use_last_img=use_last_img)
            if result < 0:
                self.wrap_success = False
                ## NO IK found, need to tune self.len
                if self.lp > 0.02: ## should always be roughly larger than the size of the gripper
                    self.lp_n = self.lp - 0.01
                    print("Next L' to test is {}".format(self.lp_n))
                    param_updated = True
                    with open("./save/log.txt", 'a') as file:
                        file.write("No IK found. Reduce L'.\n")
                else:
                    print('Safety distance between the gripper and the rod cannot be guaranteed!')
                    self.r_n = self.r - 0.005 ## try to reduce the r instead?
                    self.lp_n = 0.06
                    with open("./save/log.txt", 'a') as file:
                        file.write("No IK found. Safety distance reached. Reduce r.\n")
            else:
                self.wrap_success = True
                with open("./save/log.txt", 'a') as file:
                    file.write("Done wrap.\n")
                print("***Do one wrap successfully***")

        ## First need to manually confirm that the wrap is done successfully (no rope tangling the finger or other unexpected case).
        ## failure case will not be recorded (neither parameters nor )
        if input_option in ['new_learning', 'continue_previous']:
            len_fb = check_len(self.ic.cv_image, self.rod.info.box2d, self.rope.info.hue, self.rope.info.diameter)
            print("Extra length is: {:.1f}".format(len_fb))

            adv_fb = check_adv(self.ic.cv_image, self.rod.info.box2d, self.rope.info.hue, self.rope.info.diameter)
            print("Tested advnace is: {:.1f}%, ".format(adv_fb*100))

            wait_for_input = True
            while wait_for_input:
                choice = input('Accept this wrapping result?(Y/N)')
                if choice == 'y' or choice == 'Y':
                    wait_for_input = False
                elif choice == 'n' or choice == 'N':
                    print('Choose not to accept current wrapping result, will not use this result for learning.')
                    return
                else:
                    pass

            with open("./save/log.txt", 'a') as file:
                file.write("len_fb is {:.1f},".format(len_fb))
                file.write("adv_fb is {:.1f}%,".format(adv_fb*100))

            self.wrap_no += 1

            print("***Learning phrase...***")
            
            ## update L'
            ## len_new = len - k2*(len_feedback - threshold2)
            d_r =  0.001*(len_fb-self.rope.info.diameter*1.5) ## delta_r
            if self.last_len_fb > 0:
                # if (((last_len_fb-len_fb)/len_fb > 0.1) or len_fb > self.rope.info.diameter*1.5) and (len_fb-self.rope.info.diameter*1.5 > 0):
                # if ((d_r > 0.002) or len_fb > self.rope.info.diameter*1.5) and (len_fb-self.rope.info.diameter*1.5 > 0):
                if d_r > 0:
                    self.r_n = self.r - d_r
                    self.lp_n = 0.06 ## having a new self.r, then start to search L' from beginning
                    print("Next self.r to test is {}".format(self.r_n))
                    param_updated = True
                    self.last_len_fb = len_fb
                    with open("./save/log.txt", 'a') as file:
                        file.write("change r to {:.1f},".format(self.r_n*1000))
                    
                else:
                    print('The selection of r becomes stable')
                    param_stable[0] = True
                    with open("./save/log.txt", 'a') as file:
                        file.write("r becomes stable,")
            else:
                self.last_len_fb = len_fb
            
            ## adv_new = adv - k1*(adv_feedback - threshold1)
            # d_adv = 0.04*(adv_fb - 0.2)
            d_adv = 0.04*(adv_fb - 0.0)
            if self.last_adv_fb > 0:
                # if (abs(adv_fb-last_adv_fb)/adv_fb > 0.1) or adv_fb > 0.5:
                # if (abs(d_adv) > 0.003) or adv_fb > 0.5:
                if (adv_fb > 1e-5) and (abs(adv_fb-self.last_adv_fb)/adv_fb > 0.05):
                    self.adv_n = self.adv - d_adv
                    print("Next self.adv to test is {}".format(self.adv_n))
                    param_updated = True
                    self.last_adv_fb = adv_fb
                    with open("./save/log.txt", 'a') as file:
                        file.write("change adv to {:.1f},".format(self.adv_n*1000))
                else:
                    print('The selection of advance becomes stable')
                    param_stable[1] = True
                    if (adv_fb > 1e-5):
                        param_updated = True
                    with open("./save/log.txt", 'a') as file:
                        file.write('adv becomes stable,')
            else:
                self.last_adv_fb = adv_fb
            
            with open('./save/param.txt', 'w') as file:
                ## first  line: stable param, or [-1.0,-1.0,-1.0] (no params found yet)
                ## second line: next to test
                ## third  line: last feedback
                ## forth  line: param stable? (1 true, -1 false)
                if param_updated:
                    [self.adv_s, self.r_s, self.lp_s] = [self.adv, self.r, self.lp]

                file.write(str("{:.4f},{:.4f},{:.4f}\n".format(self.adv_s, self.r_s, self.lp_s)))
                file.write(str("{:.4f},{:.4f},{:.4f}\n".format(self.adv_n, self.r_n, self.lp_n)))
                file.write(str("{:.5f},{:.5f}\n".format(self.last_adv_fb, self.last_len_fb)))
                file.write(str("{}\n".format(param_stable)))
    
            ## one wrap/trial is done
            with open("./save/log.txt", 'a') as file:
                file.write("\n")

            if param_stable[0] and param_stable[1]:
                print("****Find the best parameters: adv: {:.1f}, r: {:.1f}, L': {:.1f}****".format(self.adv_s*1000, self.r_s*1000, self.lp_s*1000))
                print("====LEARNING END====")

        ## make a backup
        print('Backup current parameters...')
        with open('./save/param.txt', 'r') as file_in:
            with open('./save/param_'+self.time_str + '@wrap_' + str(self.wrap_no) + '.txt', 'w') as file_backup:
                file_backup.write(file_in.read())

        cv2.imwrite('./save/image_'+self.time_str+ '@wrap_' + str(self.wrap_no) + '.jpg', self.ic.cv_image)
        print('Saving parameters and the image are done')

    def step(self, no_of_wrap, center_t, r, l, advance, debug = False, execute=True, use_last_img=False):
        curve_path = self.pg.generate_nusadua(center_t, r, l, advance)

        self.pg.publish_waypoints(curve_path)

        ## based on the frame of link_7 (not the frame of the rod)
        ## z pointing toward right
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
        ## poses[2]: pushback position (away from the rope, x and y directions, all in world coordinate)
        ## poses[3]: pushback position (away from the rope, x direction)
        ## poses[4]: pushback position (away from the rope, x direction. adjust according to current rope's pose)
        ## poses[5]: pushback position (right under the rod)
        ## grab a little bit higher, then move to the starting point of the spiral (somewhere a little bit behind the rod)
        gp_pos = self.rope.gp_estimation(self.ic.cv_image, end=0, l=l-0.02, use_last_pieces=use_last_img)

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


        # poses[3] = pose_with_offset(poses[2], [0, 0, 0.06])
        poses[3] = pose_with_offset(poses[2], [0, 0, 0.08])
        poses[4] = copy.deepcopy(poses[3])
        # if no_of_wrap == 0 :
        #     poses[4].position.y = gp_pos[1] - finger_offset[2]
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

        if execute:
            # print('move closer to the rope')
            self.j_ctrl.robot_setjoint(0, js_values[0])
            rospy.sleep(1.2)
            self.j_ctrl.robot_setjoint(0, js_values[1])
            rospy.sleep(0.5)
            ## grabbing the rope
            self.gripper.l_close()
            rospy.sleep(0.8)

            # ## test the first two points of the curve
            # self.j_ctrl.exec(0, [j_traj_1[0], j_traj_1[1]], 0.2)

            # print('wrapping...')
            self.j_ctrl.exec(0, j_traj_1, 0.2)

            # print('straighten out the rope')
            ## straighten out the rope
            # self.marker.show(pt_2)
            self.j_ctrl.exec(0, j_traj_2, 0.2)

            rospy.sleep(1)
            self.gripper.l_open()

            # print('move out from the grasping pose')
            ## left grippermove to the side
            self.j_ctrl.robot_setjoint(0, js_values[2])
            rospy.sleep(0.3)

            # print('push the rope back a little bit')
            for i in [3,4,5,6,7]:
                rospy.sleep(0.3)
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
        rospy.sleep(1)
        self.gripper.r_open()
        rospy.sleep(1)
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
    menu += '2. Grasp the fixed end\n'
    menu += '3. Start a fresh new learning\n'
    menu += '4. Continue previous learning\n'
    menu += '5. Wrapping motion demo with current parameters\n'
    menu += '0. Exit\n'
    menu += 'Your input:'

    rospy.init_node('wrap_wrap', anonymous=True)
    # rate = rospy.Rate(10)
    rospy.sleep(1)

    rw = robot_wrapping()
    while run:
        choice = input(menu)
        
        if choice in ['1', '2', '3', '4', '5']:
            if choice == '1':
                ## reset the robot
                rw.reset()
            elif choice == '2':
                ## grasping the fixed end
                rw.rope_holding(0.13)
            elif choice == '3':
                ## start a new learning, tuning parameters automatically
                [os.remove('./debug/'+file) for file in os.listdir('./debug') if file.endswith('.jpg')]
                rw.wrapping('new_learning')
            elif choice == '4':
                ## continue previous learning, tuning parameters automatically
                [os.remove('./debug/'+file) for file in os.listdir('./debug') if file.endswith('.jpg')]
                param_filename = input('Checkpoint file: (a param file under ./save folder, default: param.txt)\n')
                if param_filename == '':
                    param_filename = 'param.txt'
                    print('use the default param.txt file')
                rw.wrapping('continue_previous', param_filename)
            elif choice == '5':
                ## demo the saved parameters
                param_filename = input('Checkpoint file: (a param file under ./save folder, default: param.txt)\n')
                if param_filename == '':
                    param_filename = 'param.txt'
                    print('use the default param.txt file')
                rw.wrapping('demo_current', param_filename)

        else:
            ## exit
            run = False
