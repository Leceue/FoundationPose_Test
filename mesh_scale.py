#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网格尺寸修正/单位转换工具

支持：
- 统一缩放：--scale 0.001（所有顶点乘以该系数）
- 单位转换到米：--unit_in mm --unit_out m（内置 mm/cm/inch/ft/m）
- 让最长边匹配目标长度：--fit_longest 0.1（将 AABB 的最长边缩放到 0.1m）
- 让对角线匹配目标长度：--fit_diag 0.173
- 居中到原点：--center centroid/aabb/none（默认 none）

示例：
- mm -> m 并将模型中心移到原点：
  python mesh_scale.py --in demo_data/bowl_1/mesh/bowl_1.obj --out demo_data/bowl_1/mesh/bowl_1_m.obj \
    --unit_in mm --unit_out m --center centroid

- 直接按比例缩放（从厘米到米等价于 0.01）：
  python mesh_scale.py --in model.obj --out model_scaled.obj --scale 0.01 --center aabb

- 将最长边缩放到 0.1m，且中心对齐原点：
  python mesh_scale.py --in model.obj --out model_fit.obj --fit_longest 0.1 --center aabb

注意：
- 该脚本保持 UV/法线/材质引用，trimesh 导出 OBJ 时会自动写 MTL（在输出目录）。
- 缩放为“均匀缩放”，不改变比例关系；单位转换与 fit_* 互斥，若同时传入按先后优先顺序计算并相乘。
"""

import os
import argparse
import numpy as np
import trimesh


UNIT_TO_M = {
    'm': 1.0,
    'meter': 1.0,
    'metre': 1.0,
    'mm': 1e-3,
    'millimeter': 1e-3,
    'millimetre': 1e-3,
    'cm': 1e-2,
    'centimeter': 1e-2,
    'centimetre': 1e-2,
    'inch': 0.0254,
    'in': 0.0254,
    'ft': 0.3048,
    'foot': 0.3048,
}


def compute_aabb_sizes(mesh: trimesh.Trimesh):
    mins, maxs = mesh.bounds
    sizes = maxs - mins
    longest = float(sizes.max())
    diag = float(np.linalg.norm(sizes))
    return sizes, longest, diag


def main():
    parser = argparse.ArgumentParser(description='Mesh scale/unit conversion')
    parser.add_argument('--in', dest='inp', type=str, required=True, help='输入网格（OBJ/PLY/STL 等）')
    parser.add_argument('--out', dest='out', type=str, required=True, help='输出网格（OBJ/PLY/STL 等）')

    # 缩放策略（可叠加）：scale × unit × fit
    parser.add_argument('--scale', type=float, default=None, help='统一缩放系数（例如 0.001）')
    parser.add_argument('--unit_in', type=str, default=None, help='输入单位（mm/cm/inch/ft/m）')
    parser.add_argument('--unit_out', type=str, default='m', help='输出单位（默认 m）')
    parser.add_argument('--fit_longest', type=float, default=None, help='将 AABB 最长边缩放到目标长度（米）')
    parser.add_argument('--fit_diag', type=float, default=None, help='将 AABB 对角线缩放到目标长度（米）')

    # 缩放中心
    parser.add_argument('--center', type=str, default='none', choices=['none', 'centroid', 'aabb'], help='缩放前是否将模型平移到原点')

    args = parser.parse_args()

    mesh: trimesh.Trimesh = trimesh.load(args.inp, force='mesh')
    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError('读取网格失败或不是三角网（Trimesh）')

    # 打印原始信息
    sizes0, longest0, diag0 = compute_aabb_sizes(mesh)
    print(f'[Info] input: {args.inp}')
    print(f'[Info] AABB size (x,y,z): {sizes0}, longest={longest0:.6f} m (当前单位未换算), diag={diag0:.6f} m')

    # 计算总缩放系数（统一缩放）
    total_scale = 1.0

    # 单位转换：将 unit_in → unit_out（默认 m）
    if args.unit_in is not None:
        u_in = args.unit_in.lower()
        u_out = (args.unit_out or 'm').lower()
        if u_in not in UNIT_TO_M or u_out not in UNIT_TO_M:
            raise ValueError(f'未知单位：unit_in={args.unit_in}, unit_out={args.unit_out}')
        # 把“输入单位的长度”换算成米，再换算到“输出单位”的米比例
        # 目标单位固定为米坐标系，故总缩放 = (in->m) / (out->m)。通常 out=m→ 1.0
        total_scale *= (UNIT_TO_M[u_in] / UNIT_TO_M[u_out])
        print(f'[Info] unit scale: {UNIT_TO_M[u_in]} / {UNIT_TO_M[u_out]} = {UNIT_TO_M[u_in]/UNIT_TO_M[u_out]:.9f}')

    # 直接 scale
    if args.scale is not None:
        total_scale *= float(args.scale)
        print(f'[Info] user scale: × {args.scale}')

    # 根据 AABB 拟合
    # 注意：这里用的是 AABB 尺寸，若网格旋转较大与真实长宽高不一致，可以先在 DCC 工具中对齐。
    if args.fit_longest is not None or args.fit_diag is not None:
        # 先应用到当前 mesh 的临时缩放（仅用于计算尺寸），避免“单位转换”影响未考虑
        temp = mesh.copy()
        temp.apply_scale(total_scale)
        _, longest1, diag1 = compute_aabb_sizes(temp)
        if args.fit_longest is not None:
            target = float(args.fit_longest)
            if longest1 <= 0:
                raise ValueError('最长边为 0，无法拟合')
            s_fit = target / longest1
            total_scale *= s_fit
            print(f'[Info] fit_longest: target={target} / longest={longest1} = ×{s_fit:.9f}')
        if args.fit_diag is not None:
            target = float(args.fit_diag)
            if diag1 <= 0:
                raise ValueError('对角线为 0，无法拟合')
            s_fit = target / diag1
            total_scale *= s_fit
            print(f'[Info] fit_diag: target={target} / diag={diag1} = ×{s_fit:.9f}')

    # 可选：平移到原点
    if args.center != 'none':
        if args.center == 'centroid':
            center = mesh.centroid
        elif args.center == 'aabb':
            mins, maxs = mesh.bounds
            center = (mins + maxs) * 0.5
        else:
            center = np.zeros(3)
        mesh.apply_translation(-center)
        print(f'[Info] translated to origin by -{center}')

    # 应用统一缩放（关于原点缩放）
    if abs(total_scale - 1.0) > 1e-12:
        mesh.apply_scale(total_scale)
        print(f'[Info] applied total scale × {total_scale:.9f}')

    # 导出
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(args.out)

    # 打印输出信息
    sizes2, longest2, diag2 = compute_aabb_sizes(mesh)
    print(f'[Info] output: {args.out}')
    print(f'[Info] AABB size (x,y,z): {sizes2}, longest={longest2:.6f} m, diag={diag2:.6f} m')
    print('[Done] mesh scaling completed.')


if __name__ == '__main__':
    main()
