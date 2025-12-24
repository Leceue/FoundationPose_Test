#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机械臂位姿服务节点
提供ROS服务接口来更新机械臂的位姿矩阵
"""

import rospy
import numpy as np
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Float64MultiArray


class ArmPoseService:
    def __init__(self, arm_names, pose_topic_prefix):
        """
        初始化机械臂位姿服务
        :param arm_names: 机械臂名称列表
        :param pose_topic_prefix: 位姿话题前缀
        """
        self.arm_names = arm_names
        self.pose_topic_prefix = pose_topic_prefix

        # 存储每个机械臂的位姿矩阵
        self.arm_poses = {arm_name: np.eye(4) for arm_name in arm_names}

        # 创建服务
        self.services = {}
        for arm_name in arm_names:
            service_name = f"{self.pose_topic_prefix}/{arm_name}/update_pose_raw"
            self.services[arm_name] = rospy.Service(service_name, SetBool, self._handle_update_pose)
            rospy.loginfo(f"Created service: {service_name}")

        # 创建发布者
        self.publishers = {}
        for arm_name in arm_names:
            topic_name = f"{self.pose_topic_prefix}/{arm_name}/pose_matrix"
            self.publishers[arm_name] = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10, latch=True)
            rospy.loginfo(f"Created publisher: {topic_name}")

        rospy.loginfo("ArmPoseService initialized")

    def _handle_update_pose(self, req, arm_name):
        """
        处理位姿更新请求
        :param req: SetBool请求（我们将使用data字段存储16个浮点数）
        :param arm_name: 机械臂名称
        :return: SetBool响应
        """
        try:
            # 这里使用SetBool的data字段作为示例，实际应该使用Float64MultiArray
            # 由于服务消息限制，我们将在注释中说明正确的使用方式
            rospy.logwarn("This is a simplified service interface.")
            rospy.logwarn("For full functionality, please use the Python API directly.")

            return SetBoolResponse(success=False, message="Service interface needs implementation")
        except Exception as e:
            rospy.logerr(f"Error handling service request: {e}")
            return SetBoolResponse(success=False, message=str(e))

    def update_pose(self, arm_name, pose_matrix):
        """
        更新机械臂位姿
        :param arm_name: 机械臂名称
        :param pose_matrix: 4x4位姿矩阵
        :return: 是否成功
        """
        if arm_name not in self.arm_names:
            rospy.logerr(f"Unknown arm: {arm_name}")
            return False

        if not isinstance(pose_matrix, np.ndarray) or pose_matrix.shape != (4, 4):
            rospy.logerr(f"Invalid pose matrix shape: {pose_matrix.shape}")
            return False

        # 更新存储的位姿
        self.arm_poses[arm_name] = pose_matrix

        # 发布更新后的位姿
        self._publish_pose(arm_name, pose_matrix)
        return True

    def _publish_pose(self, arm_name, pose_matrix):
        """
        发布位姿矩阵
        :param arm_name: 机械臂名称
        :param pose_matrix: 4x4位姿矩阵
        """
        try:
            msg = Float64MultiArray()
            msg.data = pose_matrix.flatten().tolist()
            self.publishers[arm_name].publish(msg)
            rospy.loginfo(f"Published updated pose for {arm_name}: {pose_matrix[:3, 3]}")
        except Exception as e:
            rospy.logerr(f"Failed to publish pose: {e}")


def main():
    """主函数"""
    rospy.init_node('arm_pose_service_node', anonymous=True)

    # 参数
    arm_names_param = rospy.get_param('~arm_names', 'arm1 arm2')
    arm_names = arm_names_param.split()
    pose_topic_prefix = rospy.get_param('~pose_topic_prefix', '/arm_pose')

    # 创建服务节点
    service_node = ArmPoseService(arm_names, pose_topic_prefix)

    rospy.loginfo("ArmPoseServiceNode is running...")
    rospy.loginfo(f"Arm names: {arm_names}")
    rospy.loginfo(f"Topic prefix: {pose_topic_prefix}")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")


if __name__ == '__main__':
    main()
