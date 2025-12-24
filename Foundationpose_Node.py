#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 节点：单帧多物体 FoundationPose register + 任务分发

通信协议（均为 std_msgs/String，内容 JSON）：
- YOLO 输入话题：[{"id":1,"name":"bowl","bbox":[x1,y1,x2,y2,x3,y3,x4,y4],"depth":0.35}, ...]
  bbox 为四点（像素或 0~1 归一化），若包含 "mask" 字段则优先使用。
- 机器人状态话题（各机器人各一）：{"exec_id":1,"name":"bowl","robot_id":"0","status":"1"}
  exec_id 表示机器人当前正在执行的 task_id。
- 任务下发话题：{"task_id":1,"name":"bowl","pose_matrix":[16 floats],"robot_id":"0","status":"1"}

流程：
1) 彩色/深度/内参同步到一帧，取最新 YOLO 检测列表（不过期）。
2) 对同帧中每个检测（按顺序）生成 mask，调用对应模型的 FoundationPose.register。
3) 依检测顺序生成任务池（队列）。
4) 两个机器人独立分发：
   - 若机器人空闲则分配队首任务。
   - 若机器人上报 exec_id 与当前分配的 task_id 一致，视为正在执行，若池中还有任务则立即推送下一个。
   - 若 exec_id 大于当前任务（机器人已完成并自增），也会尝试分配下一个。
5) 机器人执行完会自行更新 exec_id（外部逻辑负责）。
"""

import os
import sys
import json
import argparse
import threading

try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String
    import message_filters
    from cv_bridge import CvBridge
except Exception as e:  # pragma: no cover - 环境检查
    print("[Error] 需要 ROS1 Python 环境 (rospy, cv_bridge, message_filters)")
    print(e)
    sys.exit(1)

import numpy as np
import cv2
import trimesh
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import set_logging_format


class RealsenseRegisterNode:
    def __init__(
        self,
        model_names,
        mesh_files,
        debug_dir,
        color_topic,
        depth_topic,
        caminfo_topic,
        yolo_topic,
        robot1_state_topic,
        robot2_state_topic,
        robot1_output_topic,
        robot2_output_topic,
        depth_scale=0.001,
        min_depth=0.001,
        iter_register=5,
        debug_level=1,
        reset_topic='/foundationpose_reset',
    ):
        set_logging_format()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # 参数
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.iter_register = iter_register
        self.processed_once = False  # 仅处理一帧

        # 模型加载（全部预加载，按名字分类）
        if len(model_names) != len(mesh_files):
            raise ValueError('model_names 与 mesh_files 数量不一致')
        self.model_names = list(model_names)
        self.mesh_files_map = {n: f for n, f in zip(model_names, mesh_files)}
        for f in mesh_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
        self.meshes = {}
        self.ests = {}
        for name, mf in self.mesh_files_map.items():
            mesh = trimesh.load(mf)
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                 mesh=mesh, scorer=scorer, refiner=refiner,
                                 debug=debug_level, debug_dir=debug_dir)
            self.meshes[name] = mesh
            self.ests[name] = est
        rospy.loginfo(f'已加载模型: {list(self.ests.keys())}')

        # 检测缓存
        self.latest_dets = []
        self.latest_dets_time = None

        # 任务池与机器人状态
        self.task_queue = []  # 使用列表维护顺序，派发时 pop(0)
        self.next_task_id = 1
        self.assigned_task = {'0': None, '1': None}
        self.robot_exec = {'0': None, '1': None}
        self.robot_state_time = {'0': None, '1': None}

        # Debug 输出
        self.debug_dir = debug_dir
        os.makedirs(os.path.join(debug_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)
        self.save_debug = True

        # 订阅与发布
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.caminfo_topic = caminfo_topic
        self.yolo_topic = yolo_topic
        self.robot1_output_topic = robot1_output_topic
        self.robot2_output_topic = robot2_output_topic

        # 订阅realsense话题
        subs = [
            message_filters.Subscriber(self.color_topic, Image),
            message_filters.Subscriber(self.depth_topic, Image),
            message_filters.Subscriber(self.caminfo_topic, CameraInfo),
        ]
        # 同步回调
        self.ats = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.05, allow_headerless=False)
        self.ats.registerCallback(self._cb_wrapper)

        # 订阅yolo话题与机器人状态话题
        self.yolo_sub = rospy.Subscriber(self.yolo_topic, String, self._yolo_cb, queue_size=1)
        self.robot1_state_sub = rospy.Subscriber(robot1_state_topic, String, self._robot_state_cb, callback_args='0', queue_size=1)
        self.robot2_state_sub = rospy.Subscriber(robot2_state_topic, String, self._robot_state_cb, callback_args='1', queue_size=1)
        # 发布任务话题
        self.robot1_task_pub = rospy.Publisher(self.robot1_output_topic, String, queue_size=1, latch=True)
        self.robot2_task_pub = rospy.Publisher(self.robot2_output_topic, String, queue_size=1, latch=True)

        # 订阅复位话题
        self.reset_sub = rospy.Subscriber('/foundationpose_reset', String, self._reset_cb, queue_size=1)

        rospy.loginfo('RealsenseRegisterNode ready: register-only, multi-object, task dispatch enabled')

    # -------------------- 订阅回调 --------------------
    def _cb_wrapper(self, color_msg, depth_msg, caminfo_msg):
        # realsense话题回调
        self.callback(color_msg, depth_msg, caminfo_msg)

    def _reset_cb(self):
        with self.lock:
            self.processed_once = False
            self.task_queue.clear()
            self.assigned_task = {'0': None, '1': None}
            self.robot_exec = {'0': None, '1': None}
            rospy.loginfo('收到复位，状态已清空，可处理下一帧')

    def _yolo_cb(self, msg: String):
        ## yolo话题，格式为json，包含多个物体:yolo_dets: [{"id":序号, "name":物体种类, "bbox":[x1,y1,x2,y2,x3,y3,x4,y4], "depth":...}, ...]
        try:
            dets = json.loads(msg.data)
            if isinstance(dets, list):
                self.latest_dets = dets
        except Exception as e:
            rospy.logwarn(f'解析 yolo_topic 失败: {e}')

    def _robot_state_cb(self, msg: String, robot_id: str):
        # 机器人状态话题，格式为{"exec_id": 1, "name": "bowl", "robot_id": "0", "status": "1"}
        try:
            st = json.loads(msg.data)
            exec_id = st.get('exec_id', None)
            self.robot_exec[robot_id] = exec_id
            self.robot_state_time[robot_id] = rospy.Time.now()
            self._dispatch_tasks()
        except Exception as e:
            rospy.logwarn(f'解析机器人状态失败[{robot_id}]: {e}')

    # -------------------- 核心流程 --------------------
    def callback(self, color_msg, depth_msg, caminfo_msg):
        if self.processed_once:
            return
        with self.lock:
            try:
                if color_msg is None or depth_msg is None or caminfo_msg is None:
                    rospy.logwarn('缺少彩色/深度/内参，跳过帧')
                    return

                color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                color_rgb = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth = self._convert_depth(depth_cv)
                K = self._resolve_K(caminfo_msg)
                if K is None:
                    rospy.logwarn('内参为空，跳过帧')
                    return

                H, W = depth.shape[:2]
                if color_rgb.shape[:2] != (H, W):
                    color_rgb = cv2.resize(color_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                    rospy.loginfo('深度图大小与深度图不符合,调整彩色图像大小以匹配深度图')

                tasks_this_frame = []
                # 对每个检测执行 register
                for det in self.latest_dets:
                    name = det.get('name')
                    if name not in self.ests:
                        rospy.logwarn(f'未知模型名 [{name}]，跳过')
                        continue
                    mask_bool = self._det_to_mask(det, H, W)
                    if mask_bool is None:
                        rospy.logwarn(f'[{name}] mask 生成失败，跳过')
                        continue
                    pose = self.ests[name].register(K=K, rgb=color_rgb, depth=depth, ob_mask=mask_bool, iteration=self.iter_register)
                    if pose is None:
                        rospy.logwarn(f'[{name}] register 失败')
                        continue
                    pose_mat = np.array(pose, copy=False).reshape(4, 4)
                    pose_mat[3] = [0.0, 0.0, 0.0, 1.0]
                    task = {
                        'task_id': self.next_task_id,
                        'name': name,
                        'pose_matrix': pose_mat.reshape(-1).astype(float).tolist(),
                        'status': '1',
                    }
                    self.next_task_id += 1
                    tasks_this_frame.append(task)
                    if self.save_debug:
                        self._save_inputs(task['task_id'], color_rgb, depth, mask_bool, name)

                if not tasks_this_frame:
                    return

                self.task_queue.extend(tasks_this_frame)
                self._dispatch_tasks()
                # 执行完成
                self.processed_once = True

            except Exception as e:
                rospy.logerr(f'回调异常: {e}')
                import traceback
                traceback.print_exc()

    # -------------------- 工具函数 --------------------
    def _convert_depth(self, depth_cv):
        ''' 
        将深度图转换为 float32 米单位 
        因为realsense输出的深度图是 以毫米为单位的 uint16
        '''
        arr = np.array(depth_cv)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) * float(self.depth_scale)
        return arr.astype(np.float32)

    def _resolve_K(self, caminfo_msg):
        ''' 从 CameraInfo 消息中提取内参矩阵 '''
        try:
            return np.array(caminfo_msg.K, dtype=np.float32).reshape(3, 3)
        except Exception:
            return None

    def _det_to_mask(self, det, H, W):
        ''' 
        从检测结果生成二值 mask 
        白色区域为物体
        1) 优先使用显式 mask 字段
        2) 否则使用 bbox 四点生成凸多边形(通讯协议里使用了bbox)
        3) 坐标可为像素或归一化到 [0,1]
        4) 若无法生成则返回 None
        '''
        # 优先使用显式 mask
        if 'mask' in det:
            m = np.array(det['mask'], dtype=np.uint8)
            if m.shape[0] == H and m.shape[1] == W:
                return m > 0
        # 使用 bbox 生成凸多边形
        bbox = det.get('bbox', None)
        if bbox is None or len(bbox) < 8:
            rospy.logwarn('检测结果缺少 bbox 字段或格式错误')
            return None
        # 解析四点
        pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)[:4]
        if np.max(pts) <= 1.01:  # 归一化到像素
            pts[:, 0] *= W
            pts[:, 1] *= H
        pts = pts.astype(np.int32)
        mask_img = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask_img, pts.reshape(-1, 1, 2), 255)
        return mask_img > 0

    def _save_inputs(self, frame_idx, color_rgb, depth_arr, mask_bool, name):
        try:
            cv2.imwrite(os.path.join(self.debug_dir, 'color', f'{frame_idx:06d}_{name}.png'), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
            depth_base = os.path.join(self.debug_dir, 'depth', f'{frame_idx:06d}_{name}')
            np.save(depth_base + '.npy', depth_arr.astype(np.float32))
            cv2.imwrite(os.path.join(self.debug_dir, 'mask', f'{frame_idx:06d}_{name}.png'), mask_bool.astype(np.uint8) * 255)
        except Exception as e:
            rospy.logwarn(f'保存调试数据失败: {e}')

    # -------------------- 任务分发 --------------------
    def _dispatch_tasks(self):
        for rid in ['0', '1']:
            assigned = self.assigned_task.get(rid)
            exec_id = self.robot_exec.get(rid)

            # 机器人空闲，直接分配
            if assigned is None and self.task_queue:
                task = self.task_queue.pop(0)
                task['robot_id'] = rid
                self.assigned_task[rid] = task
                self._publish_task(task)
                continue

            # 机器人正在执行当前任务，若队列还有任务则提前推送下一个
            if assigned is not None and exec_id is not None:
                if exec_id == assigned['task_id'] and self.task_queue:
                    next_task = self.task_queue.pop(0)
                    next_task['robot_id'] = rid
                    self.assigned_task[rid] = next_task
                    self._publish_task(next_task)
                elif exec_id > assigned['task_id'] and self.task_queue:
                    # 机器人报告的 exec_id 已超前，视为已完成，换下一个
                    next_task = self.task_queue.pop(0)
                    next_task['robot_id'] = rid
                    self.assigned_task[rid] = next_task
                    self._publish_task(next_task)

    def _publish_task(self, task):
        msg = String()
        msg.data = json.dumps(task)
        rid = task.get('robot_id')
        if rid == '0':
            self.robot1_task_pub.publish(msg)
            rospy.loginfo(f'派发给机器人0: task_id={task["task_id"]}, name={task["name"]}')
        elif rid == '1':
            self.robot2_task_pub.publish(msg)
            rospy.loginfo(f'派发给机器人1: task_id={task["task_id"]}, name={task["name"]}')


def main():
    parser = argparse.ArgumentParser()

    # 核心输入参数:
    parser.add_argument('--model_names', nargs='+', required=True, help='模型名列表，与 mesh_files 按顺序对应')
    parser.add_argument('--mesh_files', nargs='+', required=True, help='对应的 mesh 文件路径列表')
    parser.add_argument('--debug_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'ROS_Debug'), help='调试输出目录')
    
    # 话题订阅参数:
    ## Realsense 相关话题
    parser.add_argument('--color_topic', type=str, default='/camera/color/image_raw')
    parser.add_argument('--depth_topic', type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--caminfo_topic', type=str, default='/camera/color/camera_info')

    ## yolo话题，格式为json，包含多个物体:yolo_dets: [{"id":..., "name":..., "bbox":[x1,y1,x2,y2,x3,y3,x4,y4], "depth":...}, ...]
    parser.add_argument('--yolo_topic',type=str,default='yolo_dets',help='格式为json的yolo检测结果话题')

    ## 机器人话题, 接收任务话题，格式为{"exec_id": 1, "name": "bowl", "robot_id": "0", "status": "1"}
    parser.add_argument('--robot1_state_topic', type=str, default='/0/task', help='机器人 1 任务话题')
    parser.add_argument('--robot2_state_topic', type=str, default='/1/task', help='机器人 2 任务话题')

    # 发布话题参数:
    ## 向机器人发布的话题
    ### 分别向两个机器人发布执行的任务，格式为{"task_id": 1, "name": "bowl", "pose_matrix": "4*4 T", "robot_id": "0", "status": "1"}
    parser.add_argument('--robot1_output_topic', type=str, default='/0/task', help='发布给机器人 1 的task话题')
    parser.add_argument('--robot2_output_topic', type=str, default='/1/task', help='发布给机器人 2 的task话题')

    # FoundastionPose模型参数
    parser.add_argument('--vis_topic', type=str, default='/pose_vis', help='发布可视化图像的 topic')
    parser.add_argument('--depth_scale', type=float, default=0.001)
    parser.add_argument('--iter_register', type=int, default=5, help='register 迭代次数')
    parser.add_argument('--no_save_debug', action='store_true', help='不保存调试数据')
    parser.add_argument('--debug_level', type=int, default=1, help='调试级别')
    parser.add_argument('--reset_topic', type=str, default='/foundationpose_reset', help='复位话题，收到消息后重置状态以处理下一帧')


    args = parser.parse_args()

    rospy.init_node('foundationpose_register_node', anonymous=True)
    node = RealsenseRegisterNode(
        model_names=args.model_names,
        mesh_files=args.mesh_files,
        debug_dir=args.debug_dir,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        caminfo_topic=args.caminfo_topic,
        yolo_topic=args.yolo_topic,
        robot1_state_topic=args.robot1_state_topic,
        robot2_state_topic=args.robot2_state_topic,
        robot1_output_topic=args.robot1_output_topic,
        robot2_output_topic=args.robot2_output_topic,
        depth_scale=args.depth_scale,
        iter_register=args.iter_register,
        reset_topic=args.reset_topic,
        debug_level=args.debug_level,
    )

    if args.no_save_debug:
        node.save_debug = False

    rospy.loginfo('foundationpose_register_node running...')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutdown requested')


if __name__ == '__main__':
    main()
