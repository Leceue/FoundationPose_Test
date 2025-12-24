#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用自定义数据进行位姿估计的演示脚本（单帧或序列）。

两种使用方式：
1) 单帧模式（推荐用于快速测试）：
   - 必填：--mesh_file, --rgb, --depth, 以及相机内参（--K_file 或 --fx/--fy/--cx/--cy）
   - 可选：--mask（物体前景掩码），--bbox（形如 xmin,ymin,xmax,ymax；若提供则据此生成掩码）
   - 可选：--depth_scale（深度单位换算，若深度PNG为毫米，建议 0.001）
    - 可选：--depth_colorized（若你的“深度图”其实是彩色可视化，请开启并提供 --depth_min/--depth_max 用于反色映射，注意仅近似）

2) 序列模式（用于连续帧跟踪）：
   - 必填：--mesh_file, --scene_dir（目录结构需与 YCB-Video 风格一致）

输出：
- 每帧 4x4 位姿矩阵保存至 debug/ob_in_cam/<id>.txt
- 可视化结果保存至 debug/track_vis/<id>.png（debug>=2），并窗口显示（debug>=1）

依赖：
- 已构建好的 mycpp/mycuda 扩展，PyTorch + CUDA，nvdiffrast，open3d，trimesh 等。
"""

import os
import sys
import argparse
import numpy as np
import imageio
import cv2
import trimesh

import torch


# 复用项目内的工具与估计器
from Utils import set_logging_format, draw_posed_3d_box, draw_xyz_axis
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from datareader import YcbineoatReader

try:
    import nvdiffrast.torch as dr
except Exception:
    dr = None


def parse_bbox(bbox_str, W, H):
    """解析 bbox 字符串为 (xmin, ymin, xmax, ymax)，并裁剪到图像范围。"""
    try:
        xmin, ymin, xmax, ymax = [int(v) for v in bbox_str.split(',')]
    except Exception:
        raise ValueError("--bbox 格式应为 xmin,ymin,xmax,ymax，例如: 100,80,300,260")
    xmin = max(0, min(W - 1, xmin))
    xmax = max(0, min(W - 1, xmax))
    ymin = max(0, min(H - 1, ymin))
    ymax = max(0, min(H - 1, ymax))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("--bbox 无效：xmax<=xmin 或 ymax<=ymin")
    return xmin, ymin, xmax, ymax


def build_K(args):
    """生成 3x3 内参矩阵 K。优先读取 --K_file，其次使用 --fx/--fy/--cx/--cy。"""
    if args.K_file:
        K = np.loadtxt(args.K_file).astype(np.float32)
        if K.shape != (3, 3):
            raise ValueError("--K_file 内容应为 3x3 矩阵")
        return K
    required = [args.fx, args.fy, args.cx, args.cy]
    if any(v is None for v in required):
        raise ValueError("请提供 --K_file，或同时提供 --fx --fy --cx --cy")
    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


def _decode_colorized_depth_to_metric(depth_rgb: np.ndarray, dmin: float, dmax: float, colormap: str = 'jet') -> np.ndarray:
    """将彩色可视化深度图近似还原为米制深度。
    注意：此过程是近似的，依赖你提供的可视化深度范围 [dmin, dmax]，并假设线性映射和指定的色图。
    当前仅支持 'jet'。输入为 RGB 格式。
    """
    if depth_rgb.ndim != 3 or depth_rgb.shape[2] < 3:
        raise ValueError('depth_colorized 模式下深度必须为3通道 RGB 图像')
    # 生成 Jet 调色板（BGR），长度 256
    gray = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    palette_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET).reshape(256, 3)
    # 将输入 RGB 转为 BGR 与调色板一致
    depth_bgr = cv2.cvtColor(depth_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = depth_bgr.shape[:2]
    flat = depth_bgr.reshape(-1, 3).astype(np.int16)
    pal = palette_bgr.astype(np.int16)[None, :, :]  # (1,256,3)
    # 逐像素寻找最近的调色板项（近似反色映射）
    diff = flat[:, None, :] - pal  # (N,256,3)
    dist2 = (diff * diff).sum(axis=2)  # (N,256)
    idx = dist2.argmin(axis=1).astype(np.float32)  # [0..255]
    s = idx / 255.0
    depth_m = dmin + s * (dmax - dmin)
    return depth_m.reshape(H, W).astype(np.float32)


def load_rgb_depth_mask(args):
    """加载单帧 RGB/Depth/Mask，并进行必要的预处理。返回 (color[h,w,3] uint8, depth[h,w] float32[m], mask[h,w] bool, K)。"""
    if not os.path.isfile(args.rgb):
        raise FileNotFoundError(f"RGB 文件不存在: {args.rgb}")
    if not os.path.isfile(args.depth):
        raise FileNotFoundError(f"Depth 文件不存在: {args.depth}")

    color = imageio.imread(args.rgb)
    if color.ndim == 2:
        color = np.stack([color] * 3, axis=-1)
    color = color[..., :3].astype(np.uint8)

    depth_raw = imageio.imread(args.depth)
    # 如果用户明确声明是“彩色可视化深度”，执行近似反色映射（需要 depth_min/max）
    if args.depth_colorized:
        if depth_raw.ndim != 3 or depth_raw.shape[2] < 3:
            raise ValueError('开启 --depth_colorized 但深度不是彩色图，请提供彩色可视化深度图')
        if args.depth_min is None or args.depth_max is None:
            raise ValueError('使用 --depth_colorized 时必须同时提供 --depth_min 与 --depth_max（米）')
        depth = _decode_colorized_depth_to_metric(depth_rgb=depth_raw[..., :3], dmin=float(args.depth_min), dmax=float(args.depth_max), colormap=args.depth_colormap)
        # colorized 解码已得到米制深度，此处忽略 depth_scale
    else:
        # 兼容 3/4 通道深度图：若被误保存为 RGB/RGBA，则取首通道作为灰度深度
        if depth_raw.ndim == 3 and depth_raw.shape[2] in (3, 4):
            depth_raw = depth_raw[..., 0]
        # 转为 float32，并按 depth_scale 换算到米（如 Z16 毫米 -> 0.001）
        depth = depth_raw.astype(np.float32) * float(args.depth_scale)

    H, W = depth.shape[:2]
    K = build_K(args)

    # 若 RGB 分辨率与深度分辨率不一致，则将 RGB/Mask 对齐到深度，并按比例缩放 K
    ch, cw = color.shape[:2]
    if (ch, cw) != (H, W):
        sy = H / float(ch)
        sx = W / float(cw)
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_LINEAR)
        # 缩放内参到目标分辨率（以深度为参考）
        K = K.copy().astype(np.float32)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

    # 掩码优先顺序：--mask -> --bbox -> (depth>=阈值)
    if args.mask and os.path.isfile(args.mask):
        m = imageio.imread(args.mask)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        # 若 Mask 分辨率与深度不同，做最近邻缩放
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (m > 0)
    elif args.bbox:
        xmin, ymin, xmax, ymax = parse_bbox(args.bbox, W, H)
        mask = np.zeros((H, W), dtype=bool)
        mask[ymin:ymax, xmin:xmax] = True
    else:
        # 兜底：用深度有效像素作为前景（适合纯净场景快速试验）
        mask = depth >= 0.001

    return color, depth, mask, K


def run_single(args):
    """单帧初始化配准并可视化。"""
    set_logging_format()

    # 加载网格
    mesh = trimesh.load(args.mesh_file)

    # 读取单帧数据
    color, depth, mask, K = load_rgb_depth_mask(args)

    # 调试目录
    debug_dir = args.debug_dir
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)

    # 渲染上下文
    glctx = dr.RasterizeCudaContext() if dr is not None else None

    # 初始化估计器
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=args.debug,
        debug_dir=debug_dir,
        glctx=glctx
    )

    # 执行初始化配准（返回 4x4 位姿矩阵，物体到相机）
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    # 保存位姿
    np.savetxt(os.path.join(debug_dir, 'ob_in_cam', '000000.txt'), pose.reshape(4, 4))

    # 可视化结果
    if args.debug >= 1:
        # 使用 run_demo.py 的同等可视化方式
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('pose', vis[..., ::-1])
        cv2.waitKey(500)
        if args.debug >= 2:
            imageio.imwrite(os.path.join(debug_dir, 'track_vis', '000000.png'), vis)

    print('单帧位姿估计完成，结果已保存到:', debug_dir)


def run_sequence(args):
    """序列模式：第一帧初始化，其余帧跟踪。"""
    set_logging_format()

    mesh = trimesh.load(args.mesh_file)

    debug_dir = args.debug_dir
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)

    glctx = dr.RasterizeCudaContext() if dr is not None else None

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=args.debug,
        debug_dir=debug_dir,
        glctx=glctx
    )

    reader = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        out_txt = os.path.join(debug_dir, 'ob_in_cam', f'{reader.id_strs[i]}.txt')
        np.savetxt(out_txt, pose.reshape(4, 4))

        if args.debug >= 1:
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('pose', vis[..., ::-1])
            cv2.waitKey(1)
            if args.debug >= 2:
                imageio.imwrite(os.path.join(debug_dir, 'track_vis', f'{reader.id_strs[i]}.png'), vis)

    print('序列位姿估计完成，结果已保存到:', debug_dir)


def main():
    parser = argparse.ArgumentParser(description='自定义物体位姿估计 Demo')
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # 通用参数
    parser.add_argument('--mesh_file', type=str, required=True, help='物体网格（.obj 等）')
    parser.add_argument('--debug', type=int, default=1, help='调试级别：0/1/2/3')
    parser.add_argument('--debug_dir', type=str, default=os.path.join(code_dir, 'debug_custom'), help='调试输出目录')
    parser.add_argument('--est_refine_iter', type=int, default=5, help='初始化配准迭代次数')
    parser.add_argument('--track_refine_iter', type=int, default=2, help='跟踪每帧迭代次数')

    # 单帧模式专用参数
    parser.add_argument('--rgb', type=str, help='RGB 图像路径')
    parser.add_argument('--depth', type=str, help='Depth 图像路径（PNG/TIFF 等）')
    parser.add_argument('--mask', type=str, default=None, help='前景掩码路径（可选，不提供则用 bbox 或 深度>0）')
    parser.add_argument('--bbox', type=str, default=None, help='矩形框 xmin,ymin,xmax,ymax（可选）')
    parser.add_argument('--K_file', type=str, default=None, help='3x3 内参矩阵 txt（优先使用）')
    parser.add_argument('--fx', type=float, default=None, help='焦距 fx（未提供 K_file 时必填）')
    parser.add_argument('--fy', type=float, default=None, help='焦距 fy（未提供 K_file 时必填）')
    parser.add_argument('--cx', type=float, default=None, help='主点 cx（未提供 K_file 时必填）')
    parser.add_argument('--cy', type=float, default=None, help='主点 cy（未提供 K_file 时必填）')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='深度单位缩放：若深度为毫米(mm)，设为 0.001；若为米(m)，设为 1.0')
    parser.add_argument('--depth_colorized', action='store_true', help='深度图是彩色可视化（如 RealSense 颜色化深度），需提供 --depth_min/--depth_max 近似反色映射')
    parser.add_argument('--depth_min', type=float, default=None, help='彩色深度可视化对应的最小深度（米）')
    parser.add_argument('--depth_max', type=float, default=None, help='彩色深度可视化对应的最大深度（米）')
    parser.add_argument('--depth_colormap', type=str, default='jet', choices=['jet'], help='彩色深度使用的色图（当前支持 jet）')

    # 序列模式专用参数
    parser.add_argument('--scene_dir', type=str, default=None, help='序列根目录（YCB-Video 风格）')

    args = parser.parse_args()

    # 判定模式
    single_mode = (args.rgb is not None) and (args.depth is not None)
    seq_mode = args.scene_dir is not None

    if single_mode and seq_mode:
        print('检测到同时提供了单帧与序列输入，优先采用单帧模式。')
        seq_mode = False

    if not single_mode and not seq_mode:
        parser.error('请提供单帧参数 --rgb/--depth/[--K_file 或 fx,fy,cx,cy]，或提供序列参数 --scene_dir')

    # 运行
    if single_mode:
        run_single(args)
    else:
        run_sequence(args)


if __name__ == '__main__':
    main()
