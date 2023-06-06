import cv2
import numpy as np


def xywh2xyxy(boxes: np.ndarray):
    x, y, w, h = boxes.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.column_stack((x1, y1, x2, y2)).astype(boxes.dtype)


def xywhn2xywh(boxes: np.ndarray, orig_w, orig_h):
    xn, yn, wn, hn = boxes.T
    x = xn * orig_w
    y = yn * orig_h
    w = wn * orig_w
    h = hn * orig_h
    return np.column_stack((x, y, w, h)).astype(np.int16)


def resize_pad(image: np.ndarray, h=640, w=640, fill_value=114):
    img_h, img_w, img_c = image.shape
    aspect_ratio = img_w / img_h
    if aspect_ratio > 1:
        new_img_w = w
        new_img_h = int(w / aspect_ratio)
    else:
        new_img_h = h
        new_img_w = int(h / aspect_ratio)
    resized_img = cv2.resize(image, (new_img_w, new_img_h))

    # width, height ratios
    ratio_x = new_img_w / img_w
    ratio_y = new_img_h / img_h
    padded_img = np.ones((h, w, img_c)) * fill_value
    left = (w - new_img_w) // 2
    bottom = (h - new_img_h) // 2

    padded_img[bottom : bottom + new_img_h, left : left + new_img_w] = resized_img
    pad_x = (w - new_img_w) // 2
    pad_y = (h - new_img_h) // 2
    return padded_img, (ratio_x, ratio_y), (pad_x, pad_y)
