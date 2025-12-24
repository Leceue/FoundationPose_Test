# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

# 说明：
# 本脚本演示 FoundationPose 的推理流程：
# 1) 读取目标物体的 3D 网格（.obj）与测试序列（RGB、深度、掩码、相机内参）
# 2) 在第一帧上进行位姿初始化（register）
# 3) 在随后的帧上基于上一帧姿态进行跟踪（track）
# 4) 根据 debug 级别输出可视化结果与每帧的 4x4 位姿矩阵
#
# 前置条件：
# - 请准备好 demo_data 目录或使用 --mesh_file/--test_scene_dir 指向你自己的数据。
# - 需要 CUDA、PyTorch、nvdiffrast、open3d 等依赖已正确安装，且 mycpp / mycuda 扩展已可导入。


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # 物体网格路径（默认指向 demo 的 mustard0）
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # 序列数据根目录（应包含 color/depth/mask/K 等）
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  # 初始化（register）阶段的精炼迭代次数
  parser.add_argument('--est_refine_iter', type=int, default=5)
  # 跟踪（track）阶段的每帧精炼迭代次数
  parser.add_argument('--track_refine_iter', type=int, default=2)
  # 调试级别：0=不输出，1=窗口显示投影，2=另存渲染/打分可视化，3=导出更多中间结果
  parser.add_argument('--debug', type=int, default=1)
  # 调试输出目录
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # 加载 3D 网格；支持带纹理的 .obj
  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  # 清空并准备调试目录（轨迹可视化、每帧位姿输出）
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  # 计算网格的定向包围盒（OBB）及其在中心坐标系下的 3D 盒，用于后续 2D 投影可视化
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  # 创建打分器、精炼器与 CUDA 渲染上下文
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  # 初始化 FoundationPose：内部会将网格中心化、构建旋转候选并做聚类降重、准备 mesh_tensors 等
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  # 读取序列数据读取器（YCB-Video 风格）；可从中获取 color/depth/mask/K/id_strs 等
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      # 第一帧：使用掩码进行初始化配准
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        # 导出中心化/变换后的网格与完整场景点云
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      # 之后帧：在上一帧位姿基础上进行跟踪精炼
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    # 保存当前帧的 4x4 位姿矩阵（物体到相机）
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      # 将姿态从中心化坐标还原到原网格坐标，用于叠加可视化
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      # 保存跟踪过程的渲染/叠加可视化到磁盘
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

