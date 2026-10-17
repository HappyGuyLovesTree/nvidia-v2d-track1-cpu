# Extracted _preprocess from a static method; class decorator removed, numerical body preserved.
# Source: https://github.com/mohamdev/InstantHMR/blob/5bca8ebeed0dc804189e63365a4967ffa7e5d4ae/instanthmr/inference.py
# Apache-2.0; retained paired license and notices in licenses/.
import math
import numpy as np
import cv2
INPUT_SIZE=224
CROP_EXPAND=1.2
IMAGENET_MEAN=np.array([.485,.456,.406],np.float32)
IMAGENET_STD=np.array([.229,.224,.225],np.float32)

def _preprocess(image_rgb: np.ndarray, bbox: np.ndarray, h: int, w: int) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """Square 1.2x crop, ImageNet normalise, and CLIFF conditioning."""
    x1, y1, x2, y2 = bbox.astype(float)
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cx_norm = 2.0 * (cx / w) - 1.0
    cy_norm = 2.0 * (cy / h) - 1.0
    b_scale = max(bw, bh) / max(w, h)
    cliff_cond = np.array([cx_norm, cy_norm, b_scale], dtype=np.float32)
    sq_size = max(bw, bh) * CROP_EXPAND
    half = sq_size / 2.0
    sq_x1 = cx - half
    sq_y1 = cy - half
    ix1 = int(math.floor(sq_x1))
    iy1 = int(math.floor(sq_y1))
    ix2 = int(math.ceil(sq_x1 + sq_size))
    iy2 = int(math.ceil(sq_y1 + sq_size))
    pad_left = max(0, -ix1)
    pad_top = max(0, -iy1)
    pad_right = max(0, ix2 - w)
    pad_bottom = max(0, iy2 - h)
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(w, ix2)
    src_y2 = min(h, iy2)
    patch = image_rgb[src_y1:src_y2, src_x1:src_x2]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop_224 = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    crop = crop_224.astype(np.float32) / 255.0
    crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
    crop = np.transpose(crop, (2, 0, 1)).astype(np.float32)
    return (crop, float(sq_x1), float(sq_y1), float(sq_size), cliff_cond)
