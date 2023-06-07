import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.transforms import xywhn2xywh, xywh2xyxy
from src.utils.visualization import colors


ID2NAME = {i: str(i) for i in range(10)}


def plot_bbox(
    image, bbox_xyxy, class_name, confidence, color=(255, 0, 0), txt_color=(255, 255, 255), lw=None
):
    """Visualizes a single bounding box on the image"""
    lw = lw or max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    x_min, y_min, x_max, y_max = bbox_xyxy
    txt_label = f"{class_name} {confidence:.1f}"
    p1, p2 = (x_min, y_min), (x_max, y_max)
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(txt_label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        image,
        txt_label,
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        lw / 3,
        txt_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return image


def plot_yolo_labels(image, bboxes_xywhn, class_ids, confidences, id2name=ID2NAME, plot=False):
    img = image.copy()
    H, W, C = img.shape
    if not isinstance(bboxes_xywhn, np.ndarray):
        bboxes_xywhn = np.array(bboxes_xywhn)
    bboxes_xywh = xywhn2xywh(bboxes_xywhn, W, H)
    bboxes_xyxy = xywh2xyxy(bboxes_xywh).tolist()
    for bbox, class_id, conf in zip(bboxes_xyxy, class_ids, confidences):
        class_name = id2name[class_id]
        color = colors(class_id)
        img = plot_bbox(img, bbox, class_name, conf, color)
    if plot:
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.imshow(img)
    return img
