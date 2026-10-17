# Extracted letterbox/unletterbox; unrelated demo CLI/UI omitted.
# Source: https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet/demo.py
# Apache-2.0; retained paired license and notices in licenses/.
import numpy as np
import cv2 as cv

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()
    top, left, newh, neww = (0, 0, target_size[0], target_size[1])
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = (target_size[0], int(target_size[1] / hw_scale))
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)
        else:
            newh, neww = (int(target_size[0] * hw_scale), target_size[1])
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    letterbox_scale = [top, left, newh, neww]
    return (img, letterbox_scale)

def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()
    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale
    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret
    ratioh, ratiow = (h / newh, w / neww)
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)
    return ret.astype(np.int32)
