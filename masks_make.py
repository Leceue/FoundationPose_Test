#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将输入图像的“非黑色像素”全部设为白色，黑色保持为黑色，实现二值化。

支持：
- 单张图片处理：--input /path/to/img.png --output /path/to/out.png
- 批量目录处理：--input_dir /path/to/dir --out_dir /path/to/outdir

判定“非黑色”的规则（默认，可调阈值与模式）：
- 灰度图：像素值 > thr 即视为非黑色（默认 thr=0）。
- 彩色图（BGR/RGB）：任一通道 > thr 即视为非黑色（默认 thr=0）。
- 带 Alpha 的图：可选用 alpha 通道（> alpha_thr）或与颜色联合判定（见 --mode）。

输出：
- 默认输出为单通道 uint8 二值图（0 或 255）。

示例：
	单张：
				python masks_make.py --input ./demo_data/bowl_1/masks/bowl_1_t.png \
												 --output ./demo_data/bowl_1/masks/bowl_1.png \
												 --thr 10

  批量：
		python masks_make.py --input_dir ./demo_data/kinect_driller_seq/masks \
												 --out_dir   ./demo_data/kinect_driller_seq/masks_bin \
												 --thr 10
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2


def to_binary_mask(img: np.ndarray, thr: int = 0, mode: str = "color", alpha_thr: int = 0) -> np.ndarray:
  """将任意通道图像转为单通道二值 mask（0/255）。

  参数：
    - thr: 颜色通道的黑阈值（0-255），>thr 视为非黑。
    - mode: 颜色/alpha 的判定模式：
        "color"（默认）：仅看颜色通道
        "alpha": 仅看 alpha 通道
        "alpha_or_color": alpha>alpha_thr 或 颜色>thr 任一满足即为非黑
    - alpha_thr: alpha 通道的黑阈值（0-255），>alpha_thr 视为非黑。

  规则：非黑色像素 -> 255；黑色 -> 0。
  """
  if img is None:
    raise ValueError("读取图像失败（img is None）")

  if img.ndim == 2:
    # 灰度图
    non_black = img > thr
  elif img.ndim == 3:
    h, w, c = img.shape
    if c >= 4 and mode in ("alpha", "alpha_or_color"):
      a_non_black = img[..., 3] > alpha_thr
      if mode == "alpha":
        non_black = a_non_black
      else:
        # alpha_or_color
        if c >= 3:
          c_non_black = (img[..., :3] > thr).any(axis=-1)
        else:
          c_non_black = img[..., 0] > thr
        non_black = a_non_black | c_non_black
    else:
      # 仅颜色
      if c >= 3:
        non_black = (img[..., :3] > thr).any(axis=-1)
      else:
        non_black = img[..., 0] > thr
  else:
    raise ValueError(f"不支持的图像维度: {img.shape}")

  mask = np.zeros(non_black.shape, dtype=np.uint8)
  mask[non_black] = 255
  return mask


def process_file(in_path: str, out_path: str, overwrite: bool = True, thr: int = 0, mode: str = "color", alpha_thr: int = 0) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	if (not overwrite) and os.path.exists(out_path):
		print(f"跳过（已存在）：{out_path}")
		return
	img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
	mask = to_binary_mask(img, thr=thr, mode=mode, alpha_thr=alpha_thr)
	ok = cv2.imwrite(out_path, mask)
	if not ok:
		raise RuntimeError(f"写入失败：{out_path}")
	print(f"完成：{in_path} -> {out_path}")


def process_dir(in_dir: str, out_dir: str, patterns=("*.png", "*.jpg", "*.jpeg"), overwrite: bool = True, thr: int = 0, mode: str = "color", alpha_thr: int = 0) -> None:
	os.makedirs(out_dir, exist_ok=True)
	files = []
	for p in patterns:
		files.extend(glob.glob(os.path.join(in_dir, p)))
	files = sorted(files)
	if not files:
		print(f"未在目录中找到图片：{in_dir}")
		return
	for f in files:
		rel = os.path.relpath(f, in_dir)
		out_path = os.path.join(out_dir, rel)
		out_dir_i = os.path.dirname(out_path)
		os.makedirs(out_dir_i, exist_ok=True)
		process_file(f, out_path, overwrite=overwrite, thr=thr, mode=mode, alpha_thr=alpha_thr)


def main():
	parser = argparse.ArgumentParser(description="非黑像素->白，二值化工具")
	g = parser.add_mutually_exclusive_group(required=True)
	g.add_argument("--input", type=str, help="输入图片路径")
	g.add_argument("--input_dir", type=str, help="输入目录（批量处理）")
	parser.add_argument("--output", type=str, help="输出图片路径（单张）")
	parser.add_argument("--out_dir", type=str, help="输出目录（批量）")
	parser.add_argument("--no_overwrite", action="store_true", help="如已存在则跳过，不覆盖")
	parser.add_argument("--thr", type=int, default=0, help="颜色通道阈值，>thr 视为非黑，默认 0")
	parser.add_argument("--mode", type=str, default="color", choices=["color","alpha","alpha_or_color"], help="非黑判定模式")
	parser.add_argument("--alpha_thr", type=int, default=0, help="alpha 通道阈值，>alpha_thr 视为非黑，默认 0")

	args = parser.parse_args()
	overwrite = not args.no_overwrite

	if args.input:
		if not args.output:
			parser.error("单张处理需要提供 --output 输出路径")
		process_file(args.input, args.output, overwrite=overwrite, thr=args.thr, mode=args.mode, alpha_thr=args.alpha_thr)
		return

	if args.input_dir:
		if not args.out_dir:
			parser.error("批量处理需要提供 --out_dir 输出目录")
		process_dir(args.input_dir, args.out_dir, overwrite=overwrite, thr=args.thr, mode=args.mode, alpha_thr=args.alpha_thr)
		return


if __name__ == "__main__":
	main()

