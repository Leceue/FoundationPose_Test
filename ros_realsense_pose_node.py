#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS 节点：订阅 Realsense 的 color/depth/camera_info 话题，生成 4x4 位姿矩阵并保存。

功能要点：
- 订阅 color/depth/camera_info（可选 mask topic）
- 支持可选的内参回退（--K_file 或 --fx/--fy/--cx/--cy），当 CameraInfo 不可用时使用
- 将每帧位姿保存到 debug_dir/ob_in_cam/000000.txt 等
- 可选地将可视化图像发布为 ROS 话题（--vis_topic）

运行示例（无需打包为 ROS package，前提是你在 ROS 的 Python 环境中运行）：
python3 ros_realsense_pose_node.py --mesh_file /path/to/obj.obj --depth_scale 0.001 --vis_topic /pose_vis --mask_topic /mask
"""

import os
import sys
import argparse
import threading

try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    import message_filters
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    import tf.transformations as tft
except Exception as e:
    print("[Error] rospy / cv_bridge / message_filters not available. This script must run inside a ROS1 Python environment.")
    print(e)
    sys.exit(1)

import numpy as np
import cv2
import trimesh

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import set_logging_format


class RealsensePoseNode:
    def __init__(self, mesh_file, debug_dir, color_topic, depth_topic, caminfo_topic, depth_scale=None, min_depth=0.001):
        set_logging_format()
        self.mesh_file = mesh_file
        self.debug_dir = debug_dir
        os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)
        # 新增：每帧保存的原始输入图像目录
        os.makedirs(os.path.join(debug_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, 'mask'), exist_ok=True)

        self.bridge = CvBridge()
        self.frame_idx = 0
        self.lock = threading.Lock()
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        # 上一帧投影得到的 bbox，用于在无外部 mask 时作为下一帧的 mask
        self.prev_bbox = None
        # 可选：用户提供的第一帧 mask（numpy bool array），当没有 mask topic 时作为第一帧的 mask
        self.init_mask = None
        self.init_mask_path = None

        # load mesh and estimator
        if not os.path.isfile(mesh_file):
            rospy.logerr(f'mesh_file not found: {mesh_file}')
            raise FileNotFoundError(mesh_file)
        mesh = trimesh.load(mesh_file)
        self.mesh = mesh
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug=1, debug_dir=debug_dir)

        # topics
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.caminfo_topic = caminfo_topic

        # initial subscribers (mask optional, appended later)
        self.color_sub = message_filters.Subscriber(self.color_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.caminfo_sub = message_filters.Subscriber(self.caminfo_topic, CameraInfo)
        self.subs = [self.color_sub, self.depth_sub, self.caminfo_sub]

        # approximate time synchronizer
        self.ats = message_filters.ApproximateTimeSynchronizer(self.subs, queue_size=10, slop=0.1)
        self.ats.registerCallback(self._callback_wrapper)

        # visualization publisher (optional)
        self.vis_pub = None
        # pose publisher (optional)
        self.pose_pub = None
        # 是否只发布不保存
        self.no_save = False

        # camera intrinsics fallback
        self._K_file = None
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        rospy.loginfo('[RealsensePoseNode] initialized')

    def _callback_wrapper(self, *args):
        # args: (color, depth, caminfo) or (color, depth, caminfo, mask)
        if len(args) == 3:
            return self.callback(args[0], args[1], args[2], None)
        elif len(args) == 4:
            return self.callback(args[0], args[1], args[2], args[3])
        else:
            rospy.logwarn(f'Unexpected number of synced args: {len(args)}')

    def _resolve_K(self, caminfo_msg):
        # 优先使用 camera_info；其次使用 K_file；再次使用 fx/fy/cx/cy；否则返回 None
        if caminfo_msg is not None:
            try:
                return np.array(caminfo_msg.K, dtype=np.float32).reshape(3, 3)
            except Exception:
                pass
        if self._K_file and os.path.isfile(self._K_file):
            try:
                return np.loadtxt(self._K_file).astype(np.float32)
            except Exception:
                pass
        if None not in (self._fx, self._fy, self._cx, self._cy):
            return np.array([[self._fx, 0, self._cx], [0, self._fy, self._cy], [0, 0, 1]], dtype=np.float32)
        return None

    def callback(self, color_msg: Image, depth_msg: Image, caminfo_msg: CameraInfo, mask_msg=None):
        with self.lock:
            try:
                # color -> RGB
                color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                color_rgb = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)

                # depth: 支持 uint16 或 float32
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth = depth_cv.astype(np.float32)
                # If depth msg was uint16, user should have provided depth_scale
                if hasattr(depth_msg, 'encoding') and (depth_msg.encoding.lower().startswith('16') or depth.dtype == np.uint16):
                    if self.depth_scale is not None:
                        depth = depth * float(self.depth_scale)
                    else:
                        rospy.logwarn('[RealsensePoseNode] depth is uint16 but depth_scale not provided; assuming depth in meters (unlikely).')

                # (之前的保存会在 mask 确定后进行，见后面保存逻辑)

                # K
                K = self._resolve_K(caminfo_msg)
                if K is None:
                    rospy.logerr('No valid camera intrinsics available (no CameraInfo, no K_file, no fx/fy/cx/cy). Skipping this frame.')
                    return

                H, W = depth.shape[:2]
                ch, cw = color_rgb.shape[:2]
                if (ch, cw) != (H, W):
                    sy = H / float(ch)
                    sx = W / float(cw)
                    color_rgb = cv2.resize(color_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                    K = K.copy().astype(np.float32)
                    K[0, 0] *= sx
                    K[1, 1] *= sy
                    K[0, 2] *= sx
                    K[1, 2] *= sy

                # mask
                # mask = (depth >= self.min_depth)
                # if mask_msg is not None:
                #     try:
                #         m_cv = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
                #         if m_cv.ndim == 3:
                #             m_cv = cv2.cvtColor(m_cv, cv2.COLOR_BGR2GRAY)
                #         mask = (m_cv > 0)
                #     except Exception:
                #         rospy.logwarn('Failed to convert mask_msg; keeping depth-based mask.')

                if mask_msg is not None:
                    try:
                        m_cv = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='passthrough')
                        if m_cv.ndim == 3:
                            m_gray = cv2.cvtColor(m_cv, cv2.COLOR_BGR2GRAY)
                        else:
                            m_gray = m_cv
                        mask = (m_gray > 0)
                        rospy.loginfo(f'Using provided mask_msg of shape {m_gray.shape} for frame {self.frame_idx}')
                    except Exception:
                        rospy.logwarn('Failed to convert mask_msg; falling back to depth-based mask.')
                        mask = (depth >= self.min_depth)
                else:
                    if self.frame_idx == 0:
                        # 如果用户通过 --init_mask 提供了初始 mask，则使用它
                        if getattr(self, 'init_mask', None) is not None:
                            # init_mask may have different size; resize to depth size if needed
                            im = self.init_mask
                            try:
                                if im.shape[:2] != depth.shape[:2]:
                                    im_resized = cv2.resize((im.astype(np.uint8)*255), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                                    im_bool = (im_resized > 0)
                                    # rospy.loginfo(f'Resized init_mask from {im.shape} to {depth.shape}')
                                else:
                                    im_bool = im.astype(bool)
                                    # rospy.loginfo(f'Using init_mask of shape {im.shape} as is')
                                mask = im_bool
                            except Exception as ex_init:
                                # rospy.logwarn(f'Failed to apply init_mask: {ex_init}; falling back to depth-based mask')
                                mask = (depth >= self.min_depth)
                        else:
                            mask = (depth >= self.min_depth)
                            # rospy.loginfo(f'Using depth-based mask of shape {depth.shape} for frame {self.frame_idx}')
                    else:
                        if getattr(self, 'prev_bbox', None) is not None:
                            H, W = depth.shape[:2]
                            xmin, ymin, xmax, ymax = self.prev_bbox
                            xmin = max(0, int(np.floor(xmin))); ymin = max(0, int(np.floor(ymin)))
                            xmax = min(W - 1, int(np.ceil(xmax))); ymax = min(H - 1, int(np.ceil(ymax)))
                            if xmax > xmin and ymax > ymin:
                                m = np.zeros((H, W), dtype=bool)
                                m[ymin:ymax+1, xmin:xmax+1] = True
                                mask = m
                            else:
                                mask = (depth >= self.min_depth)
                        else:
                            mask = (depth >= self.min_depth)                

                # 保存 color/depth/mask 原始输入（便于离线比对）
                # try:
                #     # color 保存为 PNG（BGR）
                #     color_save = os.path.join(self.est.debug_dir, 'color', f'{self.frame_idx:06d}.png')
                #     cv2.imwrite(color_save, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
                #     # depth 保存：若原始 depth_cv 是 uint16，保存为 16-bit png；否则保存为 .npy（float32）
                #     depth_save_base = os.path.join(self.est.debug_dir, 'depth', f'{self.frame_idx:06d}')
                #     if isinstance(depth_cv, np.ndarray) and depth_cv.dtype == np.uint16:
                #         depth_save = depth_save_base + '.png'
                #         cv2.imwrite(depth_save, depth_cv)
                #     else:
                #         depth_save = depth_save_base + '.npy'
                #         np.save(depth_save, depth_cv.astype(np.float32))
                #     # mask: 保存当前使用的 mask（mask 变量）
                #     mask_save = os.path.join(self.est.debug_dir, 'mask', f'{self.frame_idx:06d}.png')
                #     try:
                #         cv2.imwrite(mask_save, (mask.astype(np.uint8) * 255))
                #     except Exception:
                #         # fallback: try to convert and save
                #         mm = (mask.astype(np.uint8) * 255)
                #         cv2.imwrite(mask_save, mm)
                #     rospy.loginfo(f'[save] color->{color_save}, depth->{depth_save}, mask->{mask_save}')
                # except Exception as ex_save:
                #     rospy.logwarn(f'Failed to save input images for frame {self.frame_idx}: {ex_save}')
                
                # estimate
                if self.frame_idx == 0:
                    rospy.loginfo('[RealsensePoseNode] running register on first frame')
                    pose = self.est.register(K=K, rgb=color_rgb, depth=depth, ob_mask=mask, iteration=5)
                else:
                    rospy.loginfo(f'[RealsensePoseNode] tracking frame {self.frame_idx}')
                    pose = self.est.track_one(rgb=color_rgb, depth=depth, K=K, iteration=2)

                out_txt = os.path.join(self.est.debug_dir, 'ob_in_cam', f'{self.frame_idx:06d}.txt')
                # np.savetxt(out_txt, pose.reshape(4, 4))
                # rospy.loginfo(f'[RealsensePoseNode] saved pose -> {out_txt}')
                # save to file unless configured not to
                if not getattr(self, 'no_save', False):
                    np.savetxt(out_txt, pose.reshape(4, 4))
                    rospy.loginfo(f'[RealsensePoseNode] saved pose -> {out_txt}')
                else:
                    rospy.loginfo('[RealsensePoseNode] saving disabled (no_save=True)')

                # publish PoseStamped if configured
                if self.pose_pub is not None:
                    try:
                        ps = PoseStamped()
                        ps.header.stamp = color_msg.header.stamp if hasattr(color_msg, 'header') else rospy.Time.now()
                        ps.header.frame_id = color_msg.header.frame_id if hasattr(color_msg, 'header') and color_msg.header.frame_id else 'camera'
                        trans = pose[:3, 3].astype(float)
                        quat = tft.quaternion_from_matrix(pose.reshape(4, 4))
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


                # publish visualization if configured
                if self.vis_pub is not None:
                    try:
                        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
                        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
                        center_pose = pose @ np.linalg.inv(to_origin)
                        # vis = None
                        # try:
                        #     from Utils import draw_posed_3d_box, draw_xyz_axis
                        #     vis = draw_posed_3d_box(K, img=color_rgb, ob_in_cam=center_pose, bbox=bbox)
                        #     vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                        # except Exception:
                        #     vis = color_rgb
                        # vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
                        # self.vis_pub.publish(vis_msg)
                        # debug log: 内参 / bbox / 物体中心（相机坐标系）
                        rospy.loginfo(f'[vis] frame={self.frame_idx} K=[{K[0,0]:.1f},{K[1,1]:.1f},{K[0,2]:.1f},{K[1,2]:.1f}] extents={extents.tolist()} center={center_pose[:3,3].tolist()}')
                        vis = None
                        try:
                            from Utils import draw_posed_3d_box, draw_xyz_axis
                            # 保证不修改原始 color_rgb
                            vis = draw_posed_3d_box(K, img=color_rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
                            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                            if vis is None:
                                raise RuntimeError('draw_* returned None')
                        except Exception as ex:
                            rospy.logwarn(f'Visualization draw failed: {ex}. Will publish raw RGB and save debug image.')
                            vis = color_rgb.copy()
                        # 确保 uint8 RGB
                        if vis.dtype != np.uint8:
                            vis = np.clip(vis, 0, 255).astype(np.uint8)
                        # 保存一份到 debug 目录，便于离线检查
                        try:
                            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                            save_path = os.path.join(self.est.debug_dir, 'track_vis', f'vis_{self.frame_idx:06d}.png')
                            cv2.imwrite(save_path, vis_bgr)
                            rospy.loginfo(f'[vis] saved debug image -> {save_path}')
                        except Exception as ex2:
                            rospy.logwarn(f'Failed to save vis debug image: {ex2}')
                        vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
                        rospy.loginfo(f'Publishing vis frame {self.frame_idx} to topic')
                        self.vis_pub.publish(vis_msg)                        
                    except Exception as e:
                        rospy.logwarn(f'Failed to publish vis: {e}')

                self.frame_idx += 1

            except Exception as e:
                rospy.logerr(f'Error in callback: {e}')
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='ROS node to read Realsense topics and compute pose via FoundationPose')
    parser.add_argument('--mesh_file', type=str, required=True, help='物体网格文件 (.obj)')
    parser.add_argument('--debug_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'debug_custom'), help='调试输出目录')
    parser.add_argument('--color_topic', type=str, default='/camera/color/image_raw', help='彩色图像话题')
    parser.add_argument('--depth_topic', type=str, default='/camera/aligned_depth_to_color/image_raw', help='深度图话题')
    parser.add_argument('--caminfo_topic', type=str, default='/camera/color/camera_info', help='相机内参话题')
    parser.add_argument('--mask_topic', type=str, default=None, help='可选的前景掩码话题（mono8 或 bgr8）')
    parser.add_argument('--vis_topic', type=str, default=None, help='可选：发布 track 可视化图像的 topic（如 /pose_vis）')
    parser.add_argument('--K_file', type=str, default=None, help='可选：若没有 camera_info，可从 txt 加载 3x3 内参矩阵')
    parser.add_argument('--fx', type=float, default=None, help='可选：若没有 camera_info，可直接提供 fx')
    parser.add_argument('--fy', type=float, default=None, help='可选：若没有 camera_info，可直接提供 fy')
    parser.add_argument('--cx', type=float, default=None, help='可选：若没有 camera_info，可直接提供 cx')
    parser.add_argument('--cy', type=float, default=None, help='可选：若没有 camera_info，可直接提供 cy')
    parser.add_argument('--depth_scale', type=float, default=None, help='如果 depth 是 uint16，则提供深度尺度（比如 0.001 或 RealSense 的 depth_scale）')
    parser.add_argument('--min_depth', type=float, default=0.001, help='深度有效阈值（米）')
    parser.add_argument('--init_mask', type=str, default=None, help='可选：提供一张二值掩码图片文件，作为第一帧的初始 mask（当没有 mask_topic 时使用）')
    parser.add_argument('--pose_topic', type=str, default=None, help='可选：发布位姿的 topic，例如 /object_pose')
    parser.add_argument('--no_save', action='store_true', help='如果指定，则不把位姿保存为文件，仅发布（若 pose_topic 未提供则无保存也无发布）')   
    args, unknown = parser.parse_known_args()

    rospy.init_node('realsense_pose_node', anonymous=True)

    node = RealsensePoseNode(mesh_file=args.mesh_file, debug_dir=args.debug_dir, color_topic=args.color_topic, depth_topic=args.depth_topic, caminfo_topic=args.caminfo_topic, depth_scale=args.depth_scale, min_depth=args.min_depth)
    # 加载用户提供的初始 mask（若有）
    if args.init_mask:
        try:
            im = cv2.imread(args.init_mask, cv2.IMREAD_UNCHANGED)
            if im is None:
                rospy.logwarn(f'init_mask path provided but failed to load: {args.init_mask}')
            else:
                # convert to single-channel boolean mask
                if im.ndim == 3:
                    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                else:
                    im_gray = im
                mask_bool = (im_gray > 0)
                node.init_mask = mask_bool
                node.init_mask_path = args.init_mask
                rospy.loginfo(f'Loaded init_mask from {args.init_mask}, shape={mask_bool.shape}')
        except Exception as e:
            rospy.logwarn(f'Failed to load init_mask {args.init_mask}: {e}')
    # configure pose publisher / no_save
    node.no_save = args.no_save
    if args.pose_topic:
        node.pose_pub = rospy.Publisher(args.pose_topic, PoseStamped, queue_size=1)


    # 如需订阅 mask，添加到同步器并重建同步器
    if args.mask_topic:
        mask_sub = message_filters.Subscriber(args.mask_topic, Image)
        node.subs.append(mask_sub)
        node.ats = message_filters.ApproximateTimeSynchronizer(node.subs, queue_size=10, slop=0.1)
        node.ats.registerCallback(node._callback_wrapper)

    # 可选可视化发布
    if args.vis_topic:
        # node.vis_pub = rospy.Publisher(args.vis_topic, Image, queue_size=1)
        # 使用 latch=True，使得新订阅者（如后来打开的 rqt_image_view）能立刻获取最后一帧
        node.vis_pub = rospy.Publisher(args.vis_topic, Image, queue_size=1, latch=True)       

    # 保存备用内参
    node._K_file = args.K_file
    node._fx = args.fx
    node._fy = args.fy
    node._cx = args.cx
    node._cy = args.cy

    rospy.loginfo('realsense_pose_node running...')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    main()
