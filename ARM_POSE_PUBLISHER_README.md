# 机械臂位姿矩阵发布器

## 概述

这是一个ROS节点，用于向roscore发布机械臂的位姿矩阵T（4x4变换矩阵）。支持同时发布多个机械臂的位姿。

## 文件结构

- `arm_pose_publisher.py`: 主发布器节点
- `example_arm_pose.py`: 使用示例
- `arm_pose_service.py`: 服务扩展（可选）
- `pose_matrix_service.py`: 服务定义（辅助文件）

## 功能特点

1. **多机械臂支持**: 可以同时发布多个机械臂的位姿
2. **定时发布**: 可配置发布频率
3. **Python API接口**: 提供方便的API调用
4. **初始位姿加载**: 支持从文件加载初始位姿
5. **实时更新**: 支持动态更新位姿
6. **持久化发布**: 使用latch=True确保新订阅者能立即获取最新位姿

## 安装和运行

### 环境要求

- ROS 1 (已配置Python环境)
- numpy
- 其他标准Python库

### 运行方式

#### 基本运行

```bash
python3 arm_pose_publisher.py --arm_names "arm1 arm2 arm3" --publish_rate 10
```

#### 自定义话题前缀

```bash
python3 arm_pose_publisher.py --arm_names "robot_arm" --pose_topic_prefix "/my_robot" --publish_rate 5
```

#### 从文件加载初始位姿

```bash
python3 arm_pose_publisher.py --arm_names "arm1 arm2" --initial_pose_file initial_poses.txt
```

初始位姿文件格式（每行16个浮点数，对应4x4矩阵）：

```txt
1.0 0.0 0.0 0.1 0.0 1.0 0.0 0.2 0.0 0.0 1.0 0.3 0.0 0.0 0.0 1.0
0.0 0.0 1.0 0.5 1.0 0.0 0.0 0.2 0.0 1.0 0.0 0.1 0.0 0.0 0.0 1.0
```

## 订阅位姿

可以使用`rostopic echo`命令订阅位姿：

```bash
rostopic echo /arm_pose/arm1/pose_matrix
```

订阅输出示例：

```
header:
  seq: 1234
  stamp:
    secs: 1234567890
    nsecs: 123456789
  frame_id: ''
data: [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0]
```

## Python API 使用

### 基本示例

```python
import rospy
import numpy as np
from arm_pose_publisher import ArmPosePublisher

rospy.init_node('arm_pose_controller')

# 创建发布器
publisher = ArmPosePublisher(arm_names=["arm1", "arm2"], publish_rate=10)
publisher.start()

# 设置arm1的位姿
pose_arm1 = np.array([
    [1.0, 0.0, 0.0, 0.5],
    [0.0, 1.0, 0.0, 0.2],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0]
])
publisher.set_pose("arm1", pose_arm1)

# 设置arm2的位姿（带旋转）
theta = np.pi / 4
pose_arm2 = np.array([
    [np.cos(theta), -np.sin(theta), 0.0, 0.3],
    [np.sin(theta), np.cos(theta), 0.0, 0.4],
    [0.0, 0.0, 1.0, 0.8],
    [0.0, 0.0, 0.0, 1.0]
])
publisher.set_pose("arm2", pose_arm2)

# 获取当前位姿
current_pose = publisher.get_pose("arm1")
print(f"当前arm1位姿：\n{current_pose}")

# 保持运行
rospy.spin()
```

### 从文件加载

```python
import rospy
import numpy as np
from arm_pose_publisher import ArmPosePublisher

rospy.init_node('arm_pose_from_file')

publisher = ArmPosePublisher(arm_names=["robot"], publish_rate=5)
publisher.start()

# 从文件加载
initial_pose = np.loadtxt("initial_pose.txt")
publisher.set_pose("robot", initial_pose)

rospy.spin()
```

## 位姿矩阵格式

位姿矩阵T是一个4x4的齐次变换矩阵，格式如下：

```
[ R00 R01 R02 Tx ]
[ R10 R11 R12 Ty ]
[ R20 R21 R22 Tz ]
[  0   0   0   1 ]
```

其中：
- Rxx: 旋转矩阵的元素
- Tx, Ty, Tz: 平移向量的元素

## 与现有系统集成

### 与机械臂控制器集成

```python
# 假设你有一个机械臂控制器
from my_arm_controller import ArmController

arm_controller = ArmController()

# 在你的控制循环中
while not rospy.is_shutdown():
    # 获取当前机械臂位姿
    current_pose = arm_controller.get_current_pose()

    # 发布位姿
    publisher.set_pose("arm1", current_pose)

    # 处理其他逻辑
    time.sleep(0.1)
```

### 与视觉系统集成

```python
# 假设你有一个视觉系统，用于估计物体位置
from my_vision_system import VisionSystem

vision_system = VisionSystem()

# 在回调函数中
def vision_callback():
    # 获取物体在相机坐标系中的位置
    object_pose_cam = vision_system.get_object_pose()

    # 转换为机械臂基座坐标系
    object_pose_base = camera_to_base_transform @ object_pose_cam

    # 发布位姿
    publisher.set_pose("arm_target", object_pose_base)
```

## 高级功能

### 调整发布频率

```bash
python3 arm_pose_publisher.py --arm_names "arm1" --publish_rate 20
```

### 关闭发布

```python
publisher.stop()
```

### 动态添加机械臂

```python
# 注意：当前版本不支持动态添加，需要重新创建发布器
new_arm_names = publisher.arm_names + ["arm3"]
new_publisher = ArmPosePublisher(new_arm_names, publisher.pose_topic_prefix, publisher.publish_rate)
new_publisher.start()
```

## 注意事项

1. **坐标系统一致性**: 确保所有位姿使用相同的坐标系统
2. **矩阵元素顺序**: 确保矩阵元素是按行优先存储的
3. **ROS环境**: 必须在配置好ROS环境的终端中运行
4. **权限问题**: 确保对发布的话题有适当的权限
5. **服务功能**: 当前版本的服务功能需要进一步配置才能使用

## 故障排除

### 无法发布话题

- 检查ROS环境是否正确配置
- 检查话题名称是否正确
- 检查是否有足够的权限

### 订阅者无法接收消息

- 检查话题名称是否匹配
- 检查发布器是否已经启动
- 检查网络连接

### 位姿矩阵错误

- 检查矩阵维度是否为4x4
- 检查矩阵元素是否正确
- 检查坐标系统是否一致

## 未来改进

1. 完整的服务接口实现
2. 支持TF变换发布
3. 支持YAML配置文件
4. 支持动态添加/移除机械臂
5. 支持历史位姿记录
6. 支持可视化界面

## 联系方式

如有问题或建议，请联系作者。
