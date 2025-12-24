import os
import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 配置管线
pipeline = rs.pipeline()
config = rs.config()

# 可选：设置探测距离范围（单位：米）。
# 设为 None 表示不启用该阈值过滤。
# 例如：ZMIN_M = 0.28; ZMAX_M = 1.5
ZMIN_M = 0.28
ZMAX_M = 1.5

# 保持深度为640x480
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# RGB可以为任意支持的分辨率，比如1280x720
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 启动流
profile = pipeline.start(config)

# 创建对齐对象：将深度对齐到彩色（这样深度与 RGB 像素一一对应）
align_to = rs.stream.color
align = rs.align(align_to)

def intrinsics_to_K(intr):
    return np.array([[intr.fx, 0.0, intr.ppx],
                     [0.0, intr.fy, intr.ppy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

# 获取 depth_scale（用于把像素值转为米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("[Info] depth_scale:", depth_scale)

# 输出目录
out_dir = 'realsense'
os.makedirs(out_dir, exist_ok=True)
out_rgb = os.path.join(out_dir, 'rgb')
out_depth = os.path.join(out_dir, 'depth')
os.makedirs(out_rgb, exist_ok=True)
os.makedirs(out_depth, exist_ok=True)

# 保存参考流（彩色）的内参 K
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
K = intrinsics_to_K(color_stream.get_intrinsics())
np.savetxt(os.path.join(out_dir, 'cam_K.txt'), K, fmt='%.6f')
print('[Info] saved K ->', os.path.join(out_dir, 'cam_K.txt'))

try:
    # 若设置了探测距离范围，则创建阈值滤波器（单位：米）
    threshold_filter = None
    if (ZMIN_M is not None) or (ZMAX_M is not None):
        threshold_filter = rs.threshold_filter()
        if ZMIN_M is not None:
            threshold_filter.set_option(rs.option.min_distance, float(ZMIN_M))
        if ZMAX_M is not None:
            threshold_filter.set_option(rs.option.max_distance, float(ZMAX_M))
        print(f"[Info] threshold enabled: zmin={ZMIN_M}, zmax={ZMAX_M} (meters)")

    while True:
        # 等待一帧
        frames = pipeline.wait_for_frames()
        # 执行对齐
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 应用阈值滤波（超出范围的深度将置零）
        if threshold_filter is not None:
            aligned_depth_frame = threshold_filter.process(aligned_depth_frame)
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        # 转换为numpy
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        # 可选预览（不融合）：分别显示 RGB 与深度（深度简单线性拉伸）
        # 注意：实际保存的是原始 Z16 与 RGB；此处显示仅为预览
        depth_preview = depth_image
        valid = depth_preview > 0
        if np.any(valid):
            vmin = int(depth_preview[valid].min())
            vmax = int(depth_preview[valid].max())
            if vmin == vmax:
                depth_vis = np.zeros_like(depth_preview, dtype=np.uint8)
            else:
                depth_vis = ((depth_preview - vmin) / float(vmax - vmin) * 255.0).clip(0,255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_preview, dtype=np.uint8)

        cv2.imshow('Color', color_image)
        cv2.imshow('Depth(Z16)-preview', depth_vis)

        k = cv2.waitKey(1)
        if k == 27:  # ESC 保存并退出（保存一对对齐的 RGB + Z16 深度）
            stem = '0000001.png'
            cv2.imwrite(os.path.join(out_depth, stem), depth_image)  # uint16 Z16（与 color 对齐分辨率）
            cv2.imwrite(os.path.join(out_rgb, stem), color_image)    # uint8 BGR（如需 RGB 可自行转换）
            print('[Saved] rgb/depth ->', os.path.join(out_dir, '(rgb|depth)', stem))
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
