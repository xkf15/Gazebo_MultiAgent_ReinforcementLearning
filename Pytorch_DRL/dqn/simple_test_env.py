# !/usr/bin/env python  
from __future__ import print_function
import rospy
from gym_style_gazebo.srv import SimpleCtrl
from cv_bridge import CvBridge
import cv2
import numpy as np


if __name__ == '__main__':
    linear_x = 0.0
    angular_z = 0.0 
    reset = True

    rospy.wait_for_service('/gazebo_env_io/pytorch_io_service')
    pytorch_io_service = rospy.ServiceProxy('/gazebo_env_io/pytorch_io_service', SimpleCtrl)
    resp1 = pytorch_io_service(linear_x, angular_z, reset)

    print(resp1.terminal)
    print(resp1.reward)
    print(resp1.current_x)
    print(resp1.current_y)

    cv_image = CvBridge().imgmsg_to_cv2(resp1.depth_img, '32FC1')
    cv_image_resized = cv2.resize(cv_image,(128,96),interpolation=cv2.INTER_CUBIC)
    cv_image_resized = np.nan_to_num(cv_image_resized)

    #for i in range(480):
    #    for j in range(640):
    #        print(cv_image_resized[i][j], end=' ')

    cv_image_resized = 255*cv_image_resized/(np.max(cv_image_resized)-np.min(cv_image_resized))

    #cv_image_resized = 100*cv_image_resized
    #after_eq = cv2.equalizeHist(cv_image_resized)   
    cv2.imwrite("./test.jpg", cv_image_resized)
    #cv2.imshow('123', cv_image_resized)
    #cv2.waitKey()