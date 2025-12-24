#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 节点：基于每帧发布的 mask(4 个点) 和物体类别，对指定模型执行 register（不做 track），并发布位姿矩阵 T 或 Pose。

要求：在 ROS1 Python 环境中运行（包含 rospy、cv_bridge、message_filters）。
"""

import os
import sys
import argparse
import threading

try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String, Int32, Float32MultiArray
    from geometry_msgs.msg import PoseStamped, PolygonStamped
    import message_filters
    from cv_bridge import CvBridge
    import tf.transformations as tft
except Exception as e:
    print("[Error] rospy / cv_bridge / message_filters not available. Run inside ROS1 Python env.")
    print(e)
    sys.exit(1)

import numpy as np
import cv2
import trimesh
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import set_logging_format, draw_posed_3d_box, draw_xyz_axis


class RealsenseRegisterNode:
    """RealsenseRegisterNode

    中文说明：负责接收 RealSense 彩色/深度/内参以及 YOLO 提供的 mask 与类别标签，
    按照 register/track 模式调用 FoundationPose 获取位姿，并将结果发布到多个话题。
    """
    def __init__(self, model_names, mesh_files, debug_dir, color_topic, depth_topic, caminfo_topic,
                 mask_topic, class_topic, mask_type='polygon', class_type='string', depth_scale=None,
                 min_depth=0.001, iter_register=5, iter_track=2, initial_mode='register',
                 enable_track=False, auto_track=False, control_topic=None):
        """初始化节点所需资源。

        参数:
            model_names: 模型名称列表，用于标识不同物体。
            mesh_files: 与 model_names 对应的网格文件路径列表。
            debug_dir: 调试数据输出目录（保存颜色/深度/掩码/位姿）。
            color_topic/depth_topic/caminfo_topic: RealSense 相关话题名称。
            mask_topic/class_topic: YOLO mask 和类别的话题名称。
            mask_type/class_type: 指定 mask/class 的消息格式（array/polygon/image, string/int）。
            depth_scale: 当深度为 uint16 时的尺度系数。
            min_depth: 保留深度像素的最小阈值（m）。
            iter_register/iter_track: register 与 track 迭代次数，可平衡速度与精度。
            initial_mode: 启动模式（register 或 track）。
            enable_track/auto_track: 是否允许 track，以及 register 成功后是否自动切换。
            control_topic: 可选控制话题，用于外部切换模式或指定当前对象。
        """
        set_logging_format()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.iter_register = iter_register
        self.iter_track = iter_track
        self.mask_type = mask_type
        self.class_type = class_type
        self.mode = initial_mode
        self.track_enabled = enable_track
        self.auto_track = auto_track
        self.control_topic = control_topic

        # debug dirs
        self.debug_dir = debug_dir
        os.makedirs(os.path.join(debug_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)

        # model lazy management: 只在需要时加载，避免显存占用；每次仅保留一个模型
        if len(model_names) != len(mesh_files):
            raise ValueError('model_names and mesh_files must have same length')
        self.model_names = list(model_names)
        self.mesh_files_map = {n: f for n, f in zip(model_names, mesh_files)}
        for n, f in self.mesh_files_map.items():
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
        self.meshes = {}
        self.ests = {}
        self.last_pose = {}
        self.class_to_id = {name: idx for idx, name in enumerate(self.model_names)}
        self.active_model = None
        rospy.loginfo('Lazy model loading enabled; no model loaded at init')

        # topic management
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.caminfo_topic = caminfo_topic
        self.mask_topic = mask_topic
        self.class_topic = class_topic
        self.has_mask = mask_topic is not None
        self.has_class = class_topic is not None

        # publishers
        self.pose_pub = None
        self.matrix_pub = None
        self.class_pub = None
        self.id_pub = None
        self.info_pub = None
        self.vis_pub = None
        self.save_debug = True

        # subscribers & synchronizer (仅同步带时间戳的彩色/深度/camera_info，避免因无 header 的 mask/class 阻塞)
        subs = [message_filters.Subscriber(self.color_topic, Image),
                message_filters.Subscriber(self.depth_topic, Image),
                message_filters.Subscriber(self.caminfo_topic, CameraInfo)]
        self.ats = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=10,
            slop=0.1,
            allow_headerless=False  # 三个话题都有 header
        )
        self.ats.registerCallback(self._cb_wrapper)
        self.subs = subs

        # 异步缓存 mask / class（解决 YOLO 低频或无 header 导致的不同步问题）
        self.latest_mask_msg = None
        self.latest_mask_time = None
        self.latest_class_msg = None
        self.latest_class_time = None
        self.max_stale_sec = 0.5  # 可调整：允许使用最近 0.5s 内的检测来注册
        if self.has_mask:
            if mask_type == 'image':
                self.mask_sub = rospy.Subscriber(self.mask_topic, Image, self._mask_cb)
            elif mask_type == 'array':
                self.mask_sub = rospy.Subscriber(self.mask_topic, Float32MultiArray, self._mask_cb)
            else:
                self.mask_sub = rospy.Subscriber(self.mask_topic, PolygonStamped, self._mask_cb)
        if self.has_class:
            if class_type == 'int':
                self.class_sub = rospy.Subscriber(self.class_topic, Int32, self._class_cb)
            else:
                self.class_sub = rospy.Subscriber(self.class_topic, String, self._class_cb)

        # optional control channel (只订阅一次)
        self.control_sub = None
        if self.control_topic:
            self.control_sub = rospy.Subscriber(self.control_topic, String, self._control_cb)

        # frame index 初始化
        self.frame_idx = 0
        rospy.loginfo(f'RealsenseRegisterNode initialized, mode={self.mode}, track_enabled={self.track_enabled}')

    def _unload_all_except(self, keep=None):
        """卸载除 keep 外的全部已加载模型，释放显存。"""
        removed = []
        for k in list(self.ests.keys()):
            if k != keep:
                try:
                    del self.ests[k]
                    del self.meshes[k]
                    del self.last_pose[k]
                    removed.append(k)
                except Exception:
                    pass
        if removed:
            rospy.loginfo(f'Unloaded models: {removed}')
        # 尝试释放 GPU 资源
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        import gc; gc.collect()

    def _ensure_loaded(self, name):
        """若模型未加载则加载，并卸载旧模型，仅保持一个活动模型。"""
        if name in self.ests:
            # 已加载但可能不是唯一，确保清理其它
            self._unload_all_except(name)
            return
        mf = self.mesh_files_map.get(name)
        if mf is None:
            rospy.logerr(f'Model file for {name} missing in map')
            return
        try:
            mesh = trimesh.load(mf)
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                 mesh=mesh, scorer=scorer, refiner=refiner, debug=1, debug_dir=self.debug_dir)
            self.meshes[name] = mesh
            self.ests[name] = est
            self.last_pose[name] = None
            rospy.loginfo(f'Loaded model {name} (lazy) from {mf}')
        except Exception as e:
            rospy.logerr(f'Failed to load model {name}: {e}')
            return
        self._unload_all_except(name)
        self.active_model = name
        # 不在此处初始化控制话题与 frame_idx

    def _resolve_K(self, caminfo_msg):
        """根据 CameraInfo 或备用参数获取 3x3 内参矩阵。

        优先从 camera_info 中读取 K；若消息异常则返回 None，由上层决定是否跳帧。
        """
        try:
            return np.array(caminfo_msg.K, dtype=np.float32).reshape(3, 3)
        except Exception:
            return None

    # def _cb_wrapper(self, *msgs):
    #     """同步回调包装函数。

    #     message_filters 输出的消息顺序与订阅顺序一致，此处拆包并传给主 callback。
    #     """
    #     idx = 0
    #     color_msg = msgs[idx]; idx += 1
    #     depth_msg = msgs[idx]; idx += 1
    #     caminfo_msg = msgs[idx]; idx += 1
    #     mask_msg = msgs[idx] if self.has_mask else None
    #     if self.has_mask:
    #         idx += 1
    #     class_msg = msgs[idx] if self.has_class else None
    #     self.callback(color_msg, depth_msg, caminfo_msg, mask_msg, class_msg)

    def _cb_wrapper(self, color_msg, depth_msg, caminfo_msg):
        """同步回调：仅彩色、深度、内参。mask/class 使用异步最近缓存。"""
        self.callback(color_msg, depth_msg, caminfo_msg, self.latest_mask_msg, self.latest_class_msg)

    def _mask_cb(self, msg):
        self.latest_mask_msg = msg
        self.latest_mask_time = rospy.Time.now()

    def _class_cb(self, msg):
        self.latest_class_msg = msg
        self.latest_class_time = rospy.Time.now()

    def _control_cb(self, msg: String):
        """处理来自控制话题的指令，允许动态切换模式或目标对象。"""
        cmd = msg.data.strip().lower()
        if cmd.startswith('set:'):
            target = cmd.split(':', 1)[1]
            if target in self.ests:
                self.active_model = target
                rospy.loginfo(f'[control] active_model -> {target}')
            return
        if cmd == 'track':
            if self.track_enabled:
                self.mode = 'track'
                rospy.loginfo('[control] mode -> track')
            else:
                rospy.logwarn('[control] track requested but tracking disabled')
        elif cmd == 'register':
            self.mode = 'register'
            rospy.loginfo('[control] mode -> register')
        elif cmd == 'reset':
            self.active_model = None
            for k in self.last_pose:
                self.last_pose[k] = None
            rospy.loginfo('[control] reset active model and cached poses')

    def callback(self, color_msg, depth_msg, caminfo_msg, mask_msg, class_msg):
        """核心回调：完成图像同步、mask/class 解析，并执行 register 或 track。

        参数均为同步后的 ROS 消息；根据当前模式选择 register 或 track，并在成功后
        发布位姿、信息与可视化结果。"""
        with self.lock:
            try:
                if color_msg is None or depth_msg is None or caminfo_msg is None:
                    rospy.logwarn('Missing required topics; skipping frame')
                    return

                color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                color_rgb = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_arr = np.array(depth_cv)
                if depth_arr.dtype == np.uint16:
                    if self.depth_scale is None:
                        rospy.logwarn('Depth uint16 but depth_scale missing; assuming meters')
                        depth = depth_arr.astype(np.float32)
                    else:
                        depth = depth_arr.astype(np.float32) * float(self.depth_scale)
                else:
                    depth = depth_arr.astype(np.float32)

                K = self._resolve_K(caminfo_msg)
                if K is None:
                    rospy.logerr('Camera intrinsics unavailable; skipping frame')
                    return
                H, W = depth.shape[:2]
                ch, cw = color_rgb.shape[:2]
                if (ch, cw) != (H, W):
                    sy = H / float(ch)
                    sx = W / float(cw)
                    color_rgb = cv2.resize(color_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                    K = K.copy()
                    K[0, 0] *= sx; K[1, 1] *= sy
                    K[0, 2] *= sx; K[1, 2] *= sy

                idx = self.frame_idx
                tracking_now = self.track_enabled and self.mode == 'track'

                if tracking_now:
                    if not self.active_model or self.last_pose.get(self.active_model) is None:
                        rospy.logwarn('Track mode active but no registered object; waiting for detection')
                        return
                    # 确保已加载当前活动模型
                    self._ensure_loaded(self.active_model)
                    est = self.ests.get(self.active_model)
                    if est is None:
                        rospy.logwarn(f'Estimator for {self.active_model} not available')
                        return
                    rospy.loginfo(f'[{self.active_model}] track frame {idx}')
                    pose = est.track_one(rgb=color_rgb, depth=depth, K=K, iteration=self.iter_track)
                    if pose is None:
                        rospy.logwarn(f'[{self.active_model}] track failed; consider switching to register mode')
                        return
                    self.last_pose[self.active_model] = pose.reshape(4, 4)
                    self._publish_outputs(self.active_model, pose, color_msg, color_rgb, depth_arr, None, K, idx)
                    self.frame_idx += 1
                    return

                # 异步检测：检查最近缓存是否新鲜
                now_t = rospy.Time.now()
                if not self.has_mask or self.latest_mask_msg is None or \
                   (self.latest_mask_time and (now_t - self.latest_mask_time).to_sec() > self.max_stale_sec):
                    rospy.logwarn('Register mode waiting for fresh mask (async)')
                    return
                if not self.has_class or self.latest_class_msg is None or \
                   (self.latest_class_time and (now_t - self.latest_class_time).to_sec() > self.max_stale_sec):
                    rospy.logwarn('Register mode waiting for fresh class (async)')
                    return

                mask_msg = self.latest_mask_msg
                class_msg = self.latest_class_msg

                mask_bool = self._convert_mask(mask_msg, H, W)
                if mask_bool is None:
                    rospy.logwarn('Mask conversion failed; skipping frame')
                    return

                cls = self._decode_class(class_msg)
                rospy.loginfo(f'Detected class: {cls}')
                if cls not in self.ests:
                    # 懒加载该模型
                    self._ensure_loaded(cls)
                if cls not in self.ests:
                    rospy.logwarn(f'Class {cls} load failed; skipping frame')
                    return

                if self.save_debug:
                    self._save_inputs(idx, color_rgb, depth_arr, mask_bool)

                est = self.ests.get(cls)
                rospy.loginfo(f'[{cls}] register frame {idx}')
                pose = est.register(K=K, rgb=color_rgb, depth=depth, ob_mask=mask_bool, iteration=self.iter_register)
                if pose is None:
                    rospy.logwarn(f'[{cls}] register returned None')
                    return
                
                if cls in ('bowl','plate','dish','tribowl'):
                    pose4 = pose.reshape(4,4)
                    # 模型假设开口朝向模型 -Z（如不一致可改这里的 ref_dir）
                    ref_dir_obj = np.array([0,0,-1.0], dtype=np.float32)
                    # 物体坐标到相机坐标的旋转 R
                    R = pose4[:3,:3]
                    ref_dir_cam = R @ ref_dir_obj
                    # 相机视线方向设为 [0,0,1]（Z 正向为前方）；根据你相机坐标系如需改成 [0,0,-1]
                    cam_view = np.array([0,0,1.0], dtype=np.float32)
                    # 若 ref_dir_cam 与 cam_view 夹角太大（开口反向），翻转 180°
                    cosv = float(ref_dir_cam @ cam_view) / (np.linalg.norm(ref_dir_cam)+1e-6)
                    if cosv < 0:  # 或 >0，依据你的网格开口定义反向调整
                        rospy.loginfo(f'[{cls}] orientation flip applied (cos={cosv:.3f})')
                        # 选一个绕 X 轴或 Y 轴的 180° 翻转，这里绕 X
                        Rx = np.array([[1,0,0],
                                       [0,-1,0],
                                       [0,0,-1]], dtype=np.float32)
                        R_new = R @ Rx
                        pose4[:3,:3] = R_new
                        pose = pose4

                self.last_pose[cls] = pose.reshape(4, 4)
                self.active_model = cls
                if self.auto_track and self.track_enabled:
                    self.mode = 'track'

                self._publish_outputs(cls, pose, color_msg, color_rgb, depth_arr, mask_bool, K, idx)
                self.frame_idx += 1

            except Exception as e:
                rospy.logerr(f'Error in callback: {e}')
                import traceback
                traceback.print_exc()

    def _convert_mask(self, mask_msg, H, W):
        """将不同格式的 mask 消息统一转换为 HxW 的布尔数组。"""
        if mask_msg is None:
            return None
        if self.mask_type == 'image' or isinstance(mask_msg, Image):
            try:
                m_cv = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
                if m_cv.ndim == 3:
                    m_cv = cv2.cvtColor(m_cv, cv2.COLOR_BGR2GRAY)
                return m_cv > 0
            except Exception:
                return None
        if isinstance(mask_msg, Float32MultiArray) or self.mask_type == 'array':
            vals = np.array(mask_msg.data, dtype=np.float32)
            if vals.size < 8:
                return None
            pts = vals.reshape(-1, 2)[:4]
            if np.max(pts) <= 1.01:
                pts[:, 0] = pts[:, 0] * W
                pts[:, 1] = pts[:, 1] * H
            pts = pts.astype(np.int32)
            mask_img = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask_img, pts.reshape(-1, 1, 2), 255)
            return mask_img > 0
        if isinstance(mask_msg, PolygonStamped) or self.mask_type == 'polygon':
            pts = [(float(p.x), float(p.y)) for p in mask_msg.polygon.points]
            if len(pts) < 3:
                return None
            pts_np = np.array(pts, dtype=np.float32)
            if np.max(pts_np) <= 1.01:
                pts_np[:, 0] = pts_np[:, 0] * W
                pts_np[:, 1] = pts_np[:, 1] * H
            pts_np = pts_np.astype(np.int32)
            mask_img = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask_img, pts_np.reshape(-1, 1, 2), 255)
            return mask_img > 0
        return None

    def _decode_class(self, class_msg):
        """根据 class_topic 的消息类型解析出模型名称。"""
        if class_msg is None:
            return None
        if isinstance(class_msg, String):
            return class_msg.data.strip()
        if isinstance(class_msg, Int32):
            idx = int(class_msg.data)
            return self.model_names[idx] if 0 <= idx < len(self.model_names) else None
        val = getattr(class_msg, 'data', None)
        if isinstance(val, str):
            return val.strip()
        try:
            idx = int(val)
            return self.model_names[idx] if 0 <= idx < len(self.model_names) else None
        except Exception:
            return None

    def _save_inputs(self, frame_idx, color_rgb, depth_arr, mask_bool):
        """在 debug_dir 中保存当前帧的 RGB/Depth/Mask，便于离线调试。"""
        try:
            color_save = os.path.join(self.debug_dir, 'color', f'{frame_idx:06d}.png')
            cv2.imwrite(color_save, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
            depth_save_base = os.path.join(self.debug_dir, 'depth', f'{frame_idx:06d}')
            if depth_arr.dtype == np.uint16:
                cv2.imwrite(depth_save_base + '.png', depth_arr)
            else:
                np.save(depth_save_base + '.npy', depth_arr.astype(np.float32))
            mask_save = os.path.join(self.debug_dir, 'mask', f'{frame_idx:06d}.png')
            cv2.imwrite(mask_save, mask_bool.astype(np.uint8) * 255)
        except Exception as e:
            rospy.logwarn(f'Failed to save debug inputs: {e}')

    def _publish_outputs(self, cls, pose, color_msg, color_rgb, depth_arr, mask_bool, K, frame_idx):
        """统一处理所有输出：矩阵、Pose、信息话题以及可视化图像。"""
        pose_mat = np.array(pose, copy=False).reshape(4, 4)
        # 始终发布齐次变换矩阵，避免上游算法返回的最后一行缺失
        if not np.allclose(pose_mat[3], [0.0, 0.0, 0.0, 1.0]):
            pose_mat = pose_mat.copy()
            pose_mat[3] = [0.0, 0.0, 0.0, 1.0]

        if self.save_debug:
            out_txt = os.path.join(self.debug_dir, 'ob_in_cam', f'{cls}_{frame_idx:06d}.txt')
            try:
                np.savetxt(out_txt, pose_mat)
            except Exception:
                pass

        if self.matrix_pub is not None:
            try:
                ma = Float32MultiArray()
                ma.data = list(pose_mat.reshape(-1).astype(np.float32))
                self.matrix_pub.publish(ma)
            except Exception as e:
                rospy.logwarn(f'Failed to publish matrix: {e}')

        if self.pose_pub is not None:
            try:
                ps = PoseStamped()
                ps.header.stamp = color_msg.header.stamp if hasattr(color_msg, 'header') else rospy.Time.now()
                ps.header.frame_id = cls
                trans = pose_mat[:3, 3].astype(float)
                quat = tft.quaternion_from_matrix(pose_mat)
                ps.pose.position.x = float(trans[0])
                ps.pose.position.y = float(trans[1])
                ps.pose.position.z = float(trans[2])
                ps.pose.orientation.x = float(quat[0])
                ps.pose.orientation.y = float(quat[1])
                ps.pose.orientation.z = float(quat[2])
                ps.pose.orientation.w = float(quat[3])
                self.pose_pub.publish(ps)
            except Exception as e:
                rospy.logwarn(f'Failed to publish PoseStamped: {e}')

        if self.class_pub is not None:
            try:
                msg = String(); msg.data = cls
                self.class_pub.publish(msg)
            except Exception:
                pass
        if self.id_pub is not None:
            try:
                idx = self.class_to_id.get(cls, -1)
                self.id_pub.publish(Int32(data=idx))
            except Exception:
                pass
        if self.info_pub is not None:
            try:
                info = f'name:{cls},id:{self.class_to_id.get(cls,-1)},mode:{self.mode},frame:{frame_idx}'
                self.info_pub.publish(String(data=info))
            except Exception:
                pass

        # if self.vis_pub is not None:
        #     self._publish_vis(cls, color_rgb, pose_mat, K, frame_idx)
        #     self._publish_vis(cls, color_rgb, pose_mat, K, frame_idx)
        if self.vis_pub is not None:
            self._publish_vis(cls, color_rgb, pose_mat, K, frame_idx)        

    def _publish_vis(self, cls, color_rgb, pose, K, frame_idx):
        """利用 Utils 中的绘制函数生成 3D 框/坐标轴并发布图像。"""
        vis = color_rgb.copy()
        try:
            mesh = self.meshes[cls]
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        except Exception as ex:
            rospy.logwarn(f'[{cls}] visualization fallback: {ex}')
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.vis_pub.publish(vis_msg)
            if self.save_debug:
                path = os.path.join(self.debug_dir, 'track_vis', f'{cls}_{frame_idx:06d}.png')
                cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        except Exception as ex:
            rospy.logwarn(f'[{cls}] vis publish failed: {ex}')


def main():
    """命令行入口：解析参数、初始化节点与发布器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', required=True, help='模型名列表，与 mesh_files 按顺序对应')
    parser.add_argument('--mesh_files', nargs='+', required=True, help='对应的 mesh 文件路径列表')
    parser.add_argument('--debug_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'debug_custom'), help='调试输出目录')
    parser.add_argument('--color_topic', type=str, default='/camera/color/image_raw')
    parser.add_argument('--depth_topic', type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--caminfo_topic', type=str, default='/camera/color/camera_info')
    parser.add_argument('--mask_topic', type=str, required=True, help='每帧 mask 话题（来自 YOLO，四点）')
    parser.add_argument('--mask_type', type=str, choices=['image','polygon','array'], default='polygon', help='mask 话题的消息类型')
    parser.add_argument('--class_topic', type=str, required=True, help='每帧物体类别话题（String 或 Int32）')
    parser.add_argument('--class_type', type=str, choices=['string','int'], default='string', help='class_topic 的消息类型')
    parser.add_argument('--pose_topic', type=str, default='/object_pose', help='发布 PoseStamped 话题')
    parser.add_argument('--class_out_topic', type=str, default='/object_class', help='发布识别到的类别（String）')
    parser.add_argument('--id_topic', type=str, default='/object_id', help='发布物体编号 (Int32)')
    parser.add_argument('--info_topic', type=str, default='/object_info', help='发布物体信息 (String)')
    parser.add_argument('--depth_scale', type=float, default=None)
    parser.add_argument('--min_depth', type=float, default=0.001)
    parser.add_argument('--matrix_topic', type=str, default='/object_T', help='发布 4x4 位姿矩阵 (Float32MultiArray) 话题')
    parser.add_argument('--output_matrix', action='store_true', help='如果指定，则发布矩阵格式的位姿到 --matrix_topic')
    parser.add_argument('--publish_pose', action='store_true', help='如果指定，同时发布 PoseStamped 到 --pose_topic')
    parser.add_argument('--vis_topic', type=str, default='/pose_vis', help='发布可视化图像的 topic')
    parser.add_argument('--no_save_debug', action='store_true', help='如果指定，不保存 debug_dir 下的 color/depth/mask/pose')
    parser.add_argument('--iter', type=int, default=5, help='register 迭代次数')
    parser.add_argument('--track_iter', type=int, default=2, help='track_one 迭代次数')
    parser.add_argument('--mode', type=str, choices=['register','track'], default='register', help='启动模式')
    parser.add_argument('--enable_track', action='store_true', help='允许 track_one 流程')
    parser.add_argument('--auto_track', action='store_true', help='register 成功后自动切换到 track 模式')
    parser.add_argument('--control_topic', type=str, default=None, help='接收 String 控制指令 (register/track/reset/set:<name>)')
    args = parser.parse_args()

    rospy.init_node('realsense_register_node', anonymous=True)
    node = RealsenseRegisterNode(model_names=args.model_names, mesh_files=args.mesh_files, debug_dir=args.debug_dir,
                                 color_topic=args.color_topic, depth_topic=args.depth_topic, caminfo_topic=args.caminfo_topic,
                                 mask_topic=args.mask_topic, class_topic=args.class_topic, depth_scale=args.depth_scale,
                                 min_depth=args.min_depth, iter_register=args.iter, iter_track=args.track_iter,
                                 mask_type=args.mask_type, class_type=args.class_type, initial_mode=args.mode,
                                 enable_track=args.enable_track, auto_track=args.auto_track,
                                 control_topic=args.control_topic)

    # configure saving
    if args.no_save_debug:
        node.save_debug = False

    # publishers
    node.class_pub = rospy.Publisher(args.class_out_topic, String, queue_size=1, latch=True)
    node.id_pub = rospy.Publisher(args.id_topic, Int32, queue_size=1, latch=True)
    node.info_pub = rospy.Publisher(args.info_topic, String, queue_size=1, latch=True)
    if args.output_matrix:
        node.matrix_pub = rospy.Publisher(args.matrix_topic, Float32MultiArray, queue_size=1)
    if args.publish_pose:
        node.pose_pub = rospy.Publisher(args.pose_topic, PoseStamped, queue_size=1)
    if args.vis_topic:
        node.vis_pub = rospy.Publisher(args.vis_topic, Image, queue_size=1, latch=True)

    rospy.loginfo('realsense_register_node running...')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')


if __name__ == '__main__':
    main()