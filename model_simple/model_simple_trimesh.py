import trimesh

# 1. 加载OBJ文件
mesh = trimesh.load('./model_simple/bowl.obj')

# 2. 简化网格（设置目标面数，例如减少到1000个面）
target_faces = 10000
simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

# 3. 导出简化后的OBJ文件
simplified_mesh.export('./model_simple/bowl_simple.obj')


# 效果非常差劲，放弃使用trimesh进行简化
