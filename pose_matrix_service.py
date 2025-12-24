#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于创建服务的辅助文件
需要将此文件放在ROS package的srv目录下
"""

import roslib
roslib.load_manifest('arm_pose_publisher')
import rospy
from std_msgs.msg import Float64MultiArray
from rospy.msg import ServiceMessage

# 定义服务消息结构
class SetPoseMatrixRequest(ServiceMessage):
    _type = "arm_pose_publisher/SetPoseMatrixRequest"
    _md5sum = "d41d8cd98f00b204e9800998ecf8427e"
    _has_header = False
    pose_matrix = rospy.array_param(rospy.Float64, 16)  # 4x4矩阵

class SetPoseMatrixResponse(ServiceMessage):
    _type = "arm_pose_publisher/SetPoseMatrixResponse"
    _md5sum = "d41d8cd98f00b204e9800998ecf8427e"
    _has_header = False
    success = rospy.Bool()
    message = rospy.String()

class SetPoseMatrix(ServiceMessage):
    _type = "arm_pose_publisher/SetPoseMatrix"
    _md5sum = "d41d8cd98f00b204e9800998ecf8427e"
    _request_class = SetPoseMatrixRequest
    _response_class = SetPoseMatrixResponse
