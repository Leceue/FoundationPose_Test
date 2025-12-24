# ROS Pose Node Integration Guide

This document explains how the updated `ros_pose_node.py` interacts with the YOLO detection node and the RealSense streamer, what topics it consumes/publishes, and how to drive it during joint debugging.

## 1. Runtime Overview

1. RealSense node publishes synchronized RGB (`sensor_msgs/Image`), aligned depth (`sensor_msgs/Image`), and camera intrinsics (`sensor_msgs/CameraInfo`).
2. YOLO node publishes, per frame:
   - A 4-point convex mask for each detection on `--mask_topic` (default type `Float32MultiArray` with data `[x1,y1,x2,y2,x3,y3,x4,y4]`, values either pixels or normalized 0..1).
   - The object label on `--class_topic` (string or integer index, select via `--class_type`).
3. `ros_pose_node.py` synchronizes these streams, converts the mask to a binary ROI, and runs either **register** (with YOLO data) or **track** (without new detections, using FoundationPose `track_one`).
4. The node publishes pose matrices, PoseStamped messages, IDs, info strings, and a visualization image for downstream consumers.

## 2. Topics

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| Sub | `--color_topic` | `sensor_msgs/Image` (bgr8) | RGB from RealSense |
| Sub | `--depth_topic` | `sensor_msgs/Image` (16UC1 or float) | Aligned depth; pass `--depth_scale` if uint16 |
| Sub | `--caminfo_topic` | `sensor_msgs/CameraInfo` | Provides `K`; mandatory |
| Sub | `--mask_topic` | `std_msgs/Float32MultiArray` \| `geometry_msgs/PolygonStamped` \| `sensor_msgs/Image` | Select actual type via `--mask_type` (`array`=`Float32MultiArray`, `polygon`, or `image`) |
| Sub | `--class_topic` | `std_msgs/String` or `std_msgs/Int32` | Select via `--class_type`; used to pick the mesh |
| Sub (optional) | `--control_topic` | `std_msgs/String` | Runtime commands: `register`, `track`, `reset`, `set:<model>` |
| Pub | `--matrix_topic` | `std_msgs/Float32MultiArray` | 4x4 homogeneous pose matrix (row-major flatten), enable via `--output_matrix` |
| Pub | `--pose_topic` | `geometry_msgs/PoseStamped` | Enable via `--publish_pose` |
| Pub | `--class_out_topic` | `std_msgs/String` | Latest object name |
| Pub | `--id_topic` | `std_msgs/Int32` | Deterministic ID from `--model_names` order |
| Pub | `--info_topic` | `std_msgs/String` | Human-readable summary (`name:id:mode:frame`) |
| Pub | `--vis_topic` | `sensor_msgs/Image` (rgb8) | Visualization with 3D box/axes (latch enabled) |

## 3. Register vs Track Modes

- **Register mode** (default): requires YOLO mask + class each frame. Runs `FoundationPose.register`, updates the active model, and (optionally) switches to track mode automatically if `--auto_track` is set.
- **Track mode**: requires `--enable_track`. When active and an object has been registered, the node calls `FoundationPose.track_one` every frame using RGB+depth only. Mask/class topics can be absent during tracking.
- **Control topic** (String, optional):
  - `register` → force register mode.
  - `track` → switch to track mode (only if tracking enabled).
  - `reset` → clear cached poses and active model.
  - `set:<model_name>` → manually set active model when multiple meshes are loaded.

## 4. Launch Example

```bash
python3 ros_pose_node.py \
  # 使用的碗种类 bowl plant tribowl
  --model_names bowl plant tribowl \
  # 使用的碗种类 bowl plant tribowl
  --mesh_files demo_data/bowl_1/mesh/bowl.obj demo_data/bowl_1/mesh/plant.obj demo_data/bowl_1/mesh/tribowl.obj \
  # 使用的realsense话题，请务必记着开启深度rgb对齐，即align_depth:=true
  --color_topic /camera/color/image_raw \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --caminfo_topic /camera/color/camera_info \
  # 请修改为所使用的mask的话题
  --mask_topic /yolo/detection_mask \
  # 使用的mask输出格式 array是[] 或者可以用polygon模式，输出PolygonStamped格式点
  --mask_type array \
  # 请修改为所使用物体的label的话题，需要和model_names对应
  --class_topic /yolo/detection_label \
  # 种类话题格式
  --class_type string \
  # 相机深度缩放大小，一般用0.001或者标定好的0.0002500000118743628，哪个ok用哪个
  --depth_scale 0.001 \
  # 发布矩阵位姿到 matrix_topic，也发布poseStamped形式到pose_topic
  --output_matrix --publish_pose \
  # 发布的可视化位姿情况，请用rqt_image 查看
  --vis_topic /pose_vis \
  # 发布的物体信息话题
  --info_topic /object_info \
  # 发布的物体 id
  --id_topic /object_id \
  # 发布的物体的旋转矩阵话题名称，请修改或者对接为指定的话题
  --matrix_topic /object_T \
  # 使用模式 注册模式，或者注册之后跟踪模式
  --mode register --enable_track --auto_track \
  # 控制话题 可以修改两种模式，分别是注册模式和跟踪模式，请参考文档中Control topic部分
  --control_topic /pose_node/control
```

```bash
python3 ros_pose_node.py \
  --model_names bowl plate tribowl \
  --mesh_files demo_data/bowl_1/mesh/bowl.obj demo_data/bowl_1/mesh/plate.obj demo_data/bowl_1/mesh/tribowl.obj \
  --color_topic /camera/color/image_raw \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --caminfo_topic /camera/color/camera_info \
  --mask_topic /closest_object_corners \
  --mask_type array \
  --class_topic /closest_object_label \
  --class_type string \
  --depth_scale 0.001 \
  --output_matrix --publish_pose \
  --vis_topic /pose_vis \
  --info_topic /object_info \
  --class_out_topic /object_class \
  --id_topic /object_id \
  --matrix_topic /matrix_topic \
  --mode register --enable_track --auto_track \
  --control_topic /pose_node/control


python3 ros_pose_node.py \
  --model_names bowl plate tribowl \
  --mesh_files demo_data/bowl_1/mesh/bowl.obj demo_data/bowl_1/mesh/plate.obj demo_data/bowl_1/mesh/tribowl.obj \
  --color_topic /camera/color/image_raw \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --caminfo_topic /camera/color/camera_info \
  --mask_topic /closest_object_corners \
  --mask_type array \
  --class_topic /closest_object_label \
  --class_type string \
  --depth_scale 0.001 \
  --output_matrix --publish_pose \
  --vis_topic /pose_vis \
  --info_topic /object_info \
  --class_out_topic /object_class \
  --id_topic /object_id \
  --matrix_topic /matrix_topic \
  --mode register \
  --control_topic /pose_node/control


```

Typical workflow:
1. Start RealSense ROS node (color/depth/camera_info).
2. Start YOLO node publishing 4-point masks + labels.
3. Launch `ros_pose_node.py` as above. Initially in register mode, it waits for YOLO data and computes the first pose.
4. Send `std_msgs/String` with data `track` on `/pose_node/control` to freeze detection and start continuous tracking of the latest object. Use `register` to accept fresh detections again.

## 5. Pose Matrix Format

- 发布的话题 `--matrix_topic` 始终携带 16 个元素，表示相机到物体的齐次变换矩阵 `T_cam_obj`，按行主序展开。
- 第 1-3 列为旋转与平移，最后一行固定为 `[0, 0, 0, 1]`，便于直接重塑为 4x4。
- 终端快速校验：`rostopic echo /object_T -n 1 | tr -d '[],'`，确认最后四个数字为 `0 0 0 1`。
- 当需要在 Python 中使用时，可 `mat = np.array(msg.data, dtype=np.float32).reshape(4, 4)`。

## 6. Data Formats

- **Mask (`mask_type=array`)**: `Float32MultiArray.data = [x1,y1,x2,y2,x3,y3,x4,y4]`. Coordinates can be pixels or normalized (0..1). The node auto-detects normalization per frame.
- **Mask (`mask_type=polygon`)**: `PolygonStamped.polygon.points = [Point32(x,y), ...]`. Also supports normalized coords.
- **Class (`class_type=int`)**: `Int32` index into `--model_names`. (`class_type=string` expects exact mesh name.)

## 7. Outputs & Debugging

- Debug images (`debug_dir/color|depth|mask|track_vis`) and poses (`debug_dir/ob_in_cam`) are saved unless `--no_save_debug` is set.
- Visualization topic (`--vis_topic`) is latched so `rqt_image_view` can attach later.
- Matrix topic provides raw 4x4 for downstream math; Pose topic is convenient for TF or RViz.
- Use `rosnode info realsense_register_node` and `rostopic echo` for quick validation.

## 8. Joint Debugging Checklist

1. **Sync**: Verify timestamps align (message_filters window is 0.1 s). If YOLO runs slower, consider throttling RealSense or enlarging `slop`.
2. **Units**: Confirm `--depth_scale` matches RealSense depth format (usually 0.001 for Z16).
3. **Mask sanity**: Visualize YOLO mask points on the RGB image to ensure they correspond to the object.
4. **Coordinate frames**: Pose is camera→object. Convert to world frames downstream if needed.
5. **Mode toggles**: During tests, keep YOLO running so you can switch back to register quickly via the control topic.

Feel free to share this document with other developers so that both the detection team (YOLO) and the sensing team (RealSense) understand the expectations for data exchange.
