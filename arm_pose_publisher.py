#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS 节点：向roscore发布机械臂位姿矩阵T

功能要点：
- 支持发布多个机械臂的位姿矩阵
- 使用std_msgs/Float64MultiArray发布4x4矩阵 (16个元素)
- 提供服务接口供外部更新位姿
- 支持从文件加载初始位姿
- 可选的定时发布功能

运行示例：
python3 arm_pose_publisher.py --arm_names "arm1 arm2" --pose_topic_prefix "/arm_pose"

订阅示例：
rostopic echo /arm_pose/arm1/pose_matrix

更新位姿的示例代码：
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from your_service_pkg.srv import SetPoseMatrix

# 初始化ROS节点
rospy.init_node('update_arm_pose')

# 等待服务可用
rospy.wait_for_service('/arm_pose/arm1/update_pose')
try:
    update_pose = rospy.ServiceProxy('/arm_pose/arm1/update_pose', SetPoseMatrix)
    # 发送4x4位姿矩阵
    response = update_pose(pose_matrix=[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.1, 0.2, 0.3, 1.0
    ])
    print(f"Update successful: {response.success}, message: {response.message}")
except rospy.ServiceException as e:
    print(f"Service call failed: {e}")

注意：由于这不是一个完整的ROS package，服务功能需要使用Python API调用。
"""

import os
import sys
import argparse
import threading

try:
    import rospy
    from std_msgs.msg import Float64MultiArray
except Exception as e:
    print("[Error] rospy or required messages not available. This script must run inside a ROS1 Python environment.")
    print(e)
    sys.exit(1)

import numpy as np


class ArmPosePublisher:
    def __init__(self, arm_names, pose_topic_prefix, publish_rate=10.0):
        """
        初始化机械臂位姿发布器
        :param arm_names: 机械臂名称列表，如 ["arm1", "arm2"]
        :param pose_topic_prefix: 位姿话题前缀，如 "/arm_pose"
        :param publish_rate: 发布频率 (Hz)
        """
        self.arm_names = arm_names
        self.pose_topic_prefix = pose_topic_prefix
        self.publish_rate = publish_rate

        # 存储每个机械臂的位姿矩阵 (4x4 numpy array)
        self.arm_poses = {arm_name: np.eye(4) for arm_name in arm_names}

        # 创建发布者字典
        self.publishers = {}
        for arm_name in arm_names:
            topic_name = f"{self.pose_topic_prefix}/{arm_name}/pose_matrix"
            # 使用Float64MultiArray发布4x4矩阵 (16个元素)
            self.publishers[arm_name] = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10, latch=True)
            rospy.loginfo(f"Created publisher for {arm_name} on topic {topic_name}")

        # 定时发布线程
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._publish_loop)
        self.publish_thread.daemon = True

        rospy.loginfo('[ArmPosePublisher] initialized')

    def _publish_loop(self):
        """定时发布所有机械臂的位姿矩阵"""
        rate = rospy.Rate(self.publish_rate)
        while self.is_running and not rospy.is_shutdown():
            for arm_name in self.arm_names:
                self._publish_pose(arm_name)
            rate.sleep()

    def _publish_pose(self, arm_name):
        """发布单个机械臂的位姿矩阵"""
        try:
            pose_matrix = self.arm_poses[arm_name]
            # 将4x4矩阵转换为16个元素的列表
            pose_data = pose_matrix.flatten().tolist()

            # 创建Float64MultiArray消息
            msg = Float64MultiArray()
            msg.data = pose_data

            # 发布消息
            self.publishers[arm_name].publish(msg)
            rospy.logdebug(f"Published pose for {arm_name}: translation={pose_matrix[:3, 3]}, rotation={pose_matrix[:3, :3]}")
        except Exception as e:
            rospy.logwarn(f"Failed to publish pose for {arm_name}: {e}")

    def set_pose(self, arm_name, pose_matrix):
        """
        设置机械臂的位姿矩阵（Python API调用）
        :param arm_name: 机械臂名称
        :param pose_matrix: 4x4位姿矩阵 (numpy array)
        :return: 是否成功
        """
        if arm_name not in self.arm_names:
            rospy.logerr(f"Unknown arm name: {arm_name}")
            return False

        if not isinstance(pose_matrix, np.ndarray) or pose_matrix.shape != (4, 4):
            rospy.logerr(f"Invalid pose matrix shape: {pose_matrix.shape}, expected (4, 4)")
            return False

        self.arm_poses[arm_name] = pose_matrix
        # 立即发布更新后的位姿
        self._publish_pose(arm_name)
        rospy.loginfo(f"Updated pose for {arm_name}: translation={pose_matrix[:3, 3]}, rotation={pose_matrix[:3, :3]}")
        return True

    def get_pose(self, arm_name):
        """
        获取机械臂的位姿矩阵
        :param arm_name: 机械臂名称
        :return: 4x4位姿矩阵或None
        """
        if arm_name not in self.arm_names:
            rospy.logerr(f"Unknown arm name: {arm_name}")
            return None
        return self.arm_poses[arm_name]

    def start(self):
        """启动发布循环"""
        self.publish_thread.start()
        rospy.loginfo('[ArmPosePublisher] started publishing')

    def stop(self):
        """停止发布循环"""
        self.is_running = False
        if self.publish_thread.is_alive():
            self.publish_thread.join()
        rospy.loginfo('[ArmPosePublisher] stopped publishing')


def main():
    parser = argparse.ArgumentParser(description='ROS node to publish arm pose matrices')
    parser.add_argument('--arm_names', type=str, required=True,
                      help='机械臂名称列表，用空格分隔，如 "arm1 arm2"')
    parser.add_argument('--pose_topic_prefix', type=str, default='/arm_pose',
                      help='位姿话题前缀，如 "/arm_pose"')
    parser.add_argument('--publish_rate', type=float, default=10.0,
                      help='发布频率 (Hz)')
    parser.add_argument('--initial_pose_file', type=str, default=None,
                      help='可选：初始位姿文件路径，每个机械臂一行16个元素')

    args = parser.parse_args()

    # 初始化ROS节点
    rospy.init_node('arm_pose_publisher', anonymous=True)

    # 解析机械臂名称列表
    arm_names = args.arm_names.split()
    if not arm_names:
        rospy.logerr('No arm names provided!')
        return

    rospy.loginfo(f"Publishing poses for arms: {arm_names}")

    # 创建发布器
    publisher = ArmPosePublisher(arm_names, args.pose_topic_prefix, args.publish_rate)

    # 加载初始位姿（如果提供）
    if args.initial_pose_file and os.path.exists(args.initial_pose_file):
        try:
            initial_poses = np.loadtxt(args.initial_pose_file)
            if len(initial_poses.shape) == 1:
                initial_poses = initial_poses.reshape(1, -1)

            for i, arm_name in enumerate(arm_names):
                if i < initial_poses.shape[0]:
                    pose_matrix = initial_poses[i].reshape(4, 4)
                    publisher.set_pose(arm_name, pose_matrix)
                    rospy.loginfo(f"Loaded initial pose for {arm_name}: translation={pose_matrix[:3, 3]}")
        except Exception as e:
            rospy.logwarn(f"Failed to load initial poses from {args.initial_pose_file}: {e}")

    # 启动发布循环
    publisher.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
    finally:
        publisher.stop()


if __name__ == '__main__':
    main()