# Covert raw RealSense `/camera/depth/image_rect_raw` data to Open3D point cloud data
# Run this first: `roslaunch realsense2_camera rs_camera.launch`

import sys
import rospy
import tf
# import geometry_msgs.msg
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

def pose2transformation(pose):
    rot = [pose.orientation.x, pose.orientation.y,\
           pose.orientation.z, pose.orientation.w]
    ht = quaternion_matrix(rot)
    ht[:3,3] = [pose.position.x, pose.position.y, pose.position.z]
    return ht

def transformation2pose(mat):
    pose = Pose()
    q = quaternion_from_matrix(mat)
    o = mat[:3,3]
    pose.position.x = o[0]
    pose.position.y = o[1]
    pose.position.z = o[2]
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose

## workspace tf
class workspace_tf:

  def __init__(self):
    self.listener = tf.TransformListener()
    self.caster = tf.TransformBroadcaster()
    # self.rate = rate
    

  def get_tf(self, ref_frame, obj):
    ## return a homogeneous transformation matrix
    updated = False
    while updated==False:
      try:
        (trans,rot) = self.listener.lookupTransform(ref_frame, obj, rospy.Time(0))
        h = quaternion_matrix(rot)
        h[:3,3] = trans
        return h
      except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.sleep(0.1)


  def set_tf(self, ref_frame, obj, h, delay=1):
    ## h is the homogeneous transformation matrix
    updated = False
    q = quaternion_from_matrix(h)
    o = h[:3,3]
    # while updated==False:
    self.caster.sendTransform(o, q, rospy.Time.now(), obj, ref_frame)
      # rospy.sleep(delay)
      # try:
      #   _,_ = self.listener.lookupTransform(ref_frame, obj, rospy.Time(0))
      #   updated = True
      # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      #   rate.sleep()

if __name__ == '__main__':
  rospy.init_node('tf_converter', anonymous=True)
  ws_tf = workspace_tf()
  # rate = rospy.Rate(10)
  while not rospy.is_shutdown():
    ws_tf.get_tf()
    if ws_tf.tf_updated:
      print(ws_tf.trans)
      print(ws_tf.rot)
      print("====")
      # rate.sleep()
    else:
      print("marker not found")
      print("====")