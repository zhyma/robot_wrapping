import cv2, os
import numpy as np
import pickle
import copy

from scipy.stats import norm
from math import sqrt

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from cv_bridge import CvBridge
# ros
import rospy
import sensor_msgs.msg
import rospkg

from ariadne_plus.srv import getSplines, getSplinesRequest, getSplinesResponse
from scipy.interpolate import splev, splrep, splprep

import sys
sys.path.append('../../')
from utils.vision.rgb_camera import image_converter

def generateImage(img_np):
    img = Image.fromarray(img_np).convert("RGB") 
    msg = sensor_msgs.msg.Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = img.height
    msg.width = img.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * img.width
    msg.data = np.array(img).tobytes()
    return msg

def apply_crop(img, crop_corners):
    ## extract feature_map from img by using the 2d bounding box
    height = img.shape[0]
    width = img.shape[1]
    x1 = crop_corners[0,0]
    x2 = crop_corners[1,0]
    y1 = crop_corners[0,1]
    y2 = crop_corners[2,1]
    feature_map = []
    cropped_image = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(crop_corners, (ix,iy), False)
            if j > 0:
                cropped_image[iy-y1, ix-x1] = img[iy, ix]

    return cropped_image

def eval_spline(tck, crop_corners, nn_size):
    x1 = crop_corners[0,0]
    x2 = crop_corners[1,0]
    y1 = crop_corners[0,1]
    y2 = crop_corners[2,1]

    t = np.array(tck.t)
    c = np.array([tck.cx,tck.cy])
    k = int(tck.k)
    tck = [t,c,k]
    spline = splev(np.linspace(0,1,50), tck)

    x = [i*(x2-x1)/nn_size[0]+x1 for i in spline[0]]
    y = [i*(y2-y1)/nn_size[1]+y1 for i in spline[1]]
    return [x,y]

def apply_dl_mask(img, mask):
    height = img.shape[0]
    width = img.shape[1]
    resized_mask = cv2.resize(mask, (width,height))
    output = copy.deepcopy(img)
    for iy in range(height):
        for ix in range(width):
            if resized_mask[iy,ix,0] > 100:
                # print(resized_mask[iy,ix])
                output[iy, ix] = [255, 0, 0]

    return output

def check_spline(spline, box2d, l):
    ## No need to sort, I think
    on_rod = False
    cnt = 0
    while cnt < len(spline[1]):
        j = cv2.pointPolygonTest(box2d, (spline[0][cnt],spline[1][cnt]), False)
        if j > 0:
            ## on rod detected
            on_rod = True
            break

        cnt += 1

    if on_rod:
        while cnt < len(spline[1]):
            j = cv2.pointPolygonTest(box2d, (spline[0][cnt],spline[1][cnt]), False)
            if j > 0:
                cnt += 1
            else:
                ## starting from the 
                break

        acc_pixel_l = 0
        while cnt < len(spline[1])-1:
            acc_pixel_l += sqrt((spline[0][cnt]-spline[0][cnt+1])**2+(spline[1][cnt]-spline[1][cnt+1])**2)
            if acc_pixel_l > l:
                return (int(spline[0][cnt]), int(spline[1][cnt]))

            cnt += 1

        return None

    else:
        ## spline not starts from the rod, skip
        return None

def gp_estimation(img, rod_info, l=0.1, debug=False):
    ## estimating grasping point, given an image, rod's information,
    ## and expecting length of the rope (from rod to the grasping point)

    ## box2d:
    ## 3----4
    ## |    |
    ## 2----1
    sort1 = rod_info.box2d[rod_info.box2d[:,1].argsort()]
    ## upper and lower boundry
    y1 = sort1[0,1]-10
    # y1 = sort1[-2,1]-10
    y2 = img.shape[0]
    sort2 = rod_info.box2d[rod_info.box2d[:,0].argsort()]
    ## left and right boundry
    x1 = sort2[0,0]-10
    x2 = sort2[-1,0]+10

    if (x2-x1)/(y2-y1) > 640/480:
        ## too rectangle
        y1 = int(y2-(x2-x1)*480/640)
    else:
        ## too square
        xc = (x2+x1)/2
        width = (y2-y1)*640/480
        x1 = int(xc-width/2)
        x2 = int(xc+width/2)

    crop_corners= np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


    # print(crop_corners)
    ## crop to get the workspace
    cropped = apply_crop(img, crop_corners)

    bridge = CvBridge()

    input_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (640,480)) # resize necessary for the network model
    img_msg = generateImage(input_img)

    rospy.wait_for_service('get_splines')
    try:
        get_cable = rospy.ServiceProxy('get_splines', getSplines)
        req = getSplinesRequest()
        req.input_image = img_msg
        resp1 = get_cable(req)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    # print("get cable:")
    # print(resp1.tck)
    dl_mask = bridge.imgmsg_to_cv2(resp1.mask_image, desired_encoding='passthrough')

    masked = apply_dl_mask(cropped, dl_mask)

    ## img.shape[1]: x/width
    ## img.shape[0]: y/height

    x3_p = rod_info.box2d[2][0]
    x4_p = rod_info.box2d[3][0]
    y3_p = rod_info.box2d[2][1]
    y4_p = rod_info.box2d[3][1]
    l_pixel = sqrt((x3_p-x4_p)**2+(y3_p-y4_p)**2)
    scale = rod_info.l/l_pixel
    print("phsical to pixel scale is: %f/%f=%f"%(rod_info.l, l_pixel, scale))

    if debug:
        fig = plt.figure(figsize=(12,10))
        ax0 = plt.subplot2grid((2,2),(0,0))
        ax1 = plt.subplot2grid((2,2),(0,1))
        ax2 = plt.subplot2grid((2,2),(1,0),colspan=2)

        ax0.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        ax1.imshow(cv2.cvtColor(dl_mask, cv2.COLOR_BGR2RGB))

        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i in range(len(resp1.tck)):
            spline = eval_spline(resp1.tck[i], crop_corners, (640,480))
            ax2.plot(spline[0], spline[1], color='c', linewidth=2)

        plt.tight_layout()
        plt.show()

    checked = None
    for i in range(len(resp1.tck)):
        spline = eval_spline(resp1.tck[i], crop_corners, (640,480))
        checked = check_spline(spline, rod_info.box2d, l/scale)
        if checked is None:
            continue
        else:
            break

    if checked is None:
        print("No possible rope end is found")
        return None
    else:
        ## return estimated grasping point position
        # center of the rectangle, in pixel
        xc_p = (rod_info.box2d[2][0] + rod_info.box2d[0][0])/2
        yc_p = (rod_info.box2d[2][1] + rod_info.box2d[0][1])/2

        dx_p = checked[0] - xc_p
        dy_p = checked[1] - yc_p

        # estimate distance, actual, measured in meters
        dy = dx_p * scale
        dz = -dy_p * scale

        x = rod_info.pose.position.x + rod_info.r
        y = rod_info.pose.position.y + dy
        z = rod_info.pose.position.z + dz
        print("found grasping point: %.3f, %.3f, %.3f"%(x, y, z))
        return [x, y, z]


if __name__ == '__main__': 
    with open('rod_info.pickle', 'rb') as handle:
        rod_info = pickle.load(handle)

    # img = cv2.imread('image1.jpg')
    ic = image_converter()
    rospy.init_node('ariadne_test', anonymous=True)
    rospy.sleep(1)
    while ic.has_data==False:
            print('waiting for RGB data')
            rospy.sleep(0.1)

    serial_number = 0
    gp_estimation(ic.cv_image, rod_info, 0.1)
    # for i in range(30):
    #     main(ic.cv_image, rod_info.box2d, 0.1)
    #     input("test: "+str(i))