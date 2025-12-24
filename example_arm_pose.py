#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂位姿发布示例
"""

import rospy
import numpy as np
from arm_pose_publisher import ArmPosePublisher


def example_usage():
    """示例：如何使用机械臂位姿发布器"""

    # 初始化ROS节点
    rospy.init_node('arm_pose_example', anonymous=True)

    # 机械臂名称列表
    arm_names = ["arm1", "arm2", "arm3"]

    # 创建发布器
    publisher = ArmPosePublisher(arm_names=arm_names,
                               pose_topic_prefix="/industrial_arm",
                               publish_rate=5.0)  # 5 Hz

    # 启动发布循环
    publisher.start()

    rospy.loginfo("机械臂位姿发布器已启动")

    # 等待ROS节点初始化完成
    rospy.sleep(1.0)

    # 为arm1设置一个新的位姿
    # 位置：x=0.5, y=0.2, z=1.0
    # 姿态：单位矩阵（无旋转）
    pose_arm1 = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    publisher.set_pose("arm1", pose_arm1)

    # 为arm2设置一个带有旋转的位姿
    # 绕Z轴旋转45度
    theta = np.pi / 4
    pose_arm2 = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 0.3],
        [np.sin(theta), np.cos(theta), 0.0, 0.4],
        [0.0, 0.0, 1.0, 0.8],
        [0.0, 0.0, 0.0, 1.0]
    ])
    publisher.set_pose("arm2", pose_arm2)

    # 保持运行一段时间
    rospy.loginfo("保持发布5秒...")
    rospy.sleep(5.0)

    # 更新arm1的位姿
    pose_arm1_new = np.array([
        [1.0, 0.0, 0.0, 0.6],
        [0.0, 1.0, 0.0, 0.25],
        [0.0, 0.0, 1.0, 1.1],
        [0.0, 0.0, 0.0, 1.0]
    ])
    publisher.set_pose("arm1", pose_arm1_new)

    rospy.loginfo("再保持发布5秒...")
    rospy.sleep(5.0)

    # 停止发布器
    publisher.stop()
    rospy.loginfo("机械臂位姿发布器已停止")


def example_from_file():
    """示例：从文件加载初始位姿"""

    rospy.init_node('arm_pose_from_file_example', anonymous=True)

    arm_names = ["robot_arm"]

    # 创建发布器，不指定初始位姿文件
    publisher = ArmPosePublisher(arm_names=arm_names,
                               pose_topic_prefix="/robot",
                               publish_rate=10.0)
    publisher.start()

    rospy.sleep(1.0)

    # 创建初始位姿文件内容
    initial_pose = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, -1.0, 0.5],
        [0.0, 1.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 保存到临时文件
    temp_file = "/tmp/initial_pose.txt"
    np.savetxt(temp_file, initial_pose)

    rospy.loginfo(f"从文件加载初始位姿：{temp_file}")

    # 从文件加载并设置
    try:
        loaded_pose = np.loadtxt(temp_file)
        publisher.set_pose("robot_arm", loaded_pose)
        rospy.loginfo(f"成功加载位姿：\n{loaded_pose}")
    except Exception as e:
        rospy.logerr(f"加载位姿失败：{e}")

    # 清理临时文件
    import os
    os.remove(temp_file)

    # 保持运行
    rospy.loginfo("保持运行10秒...")
    rospy.sleep(10.0)

    publisher.stop()


if __name__ == '__main__':
    try:
        # 运行第一个示例
        # example_usage()

        # 运行第二个示例
        example_from_file()

    except rospy.ROSInterruptException:
        pass
