import cv2
import numpy as np
import os

im = cv2.imread('./demo_data/bowl_1/masks/color_1.png', cv2.IMREAD_UNCHANGED)

if im.shape[:2] != depth.shape[:2]:
    im_resized = cv2.resize((im.astype(np.uint8)*255), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    im_bool = (im_resized > 0)
else:
    im_bool = im.astype(bool)

im_bool = im.astype(bool)
mask = im_bool


mask_save = './test_mask.png'
try:
    cv2.imwrite(mask_save, (mask.astype(np.uint8) * 255))
except Exception:
    # fallback: try to convert and save
    mm = (mask.astype(np.uint8) * 255)
    cv2.imwrite(mask_save, mm)