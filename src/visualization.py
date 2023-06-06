import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np
from src.transforms import xywhn2xywh, xywh2xyxy


def get_matplotlib_palette(name: str = "tab10"):
    # Get the default color map
    cmap = plt.get_cmap(name)
    num_colors = cmap.N
    rgb_values = [[int(c * 255) for c in mcolors.to_rgb(cmap(i))] for i in range(num_colors)]
    return rgb_values


ID2COLOR = {label: rgb for label, rgb in zip(list(range(10)), get_matplotlib_palette())}
ID2NAME = {i: str(i) for i in range(10)}

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def plot_bbox(
    image, bbox_xyxy, class_name, confidence, color=BOX_COLOR, txt_color=TEXT_COLOR, thickness=1
):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox_xyxy
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    txt_label = f"{class_name} {confidence:.1f}"

    ((text_width, text_height), _) = cv2.getTextSize(txt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

    txt_rect_y_min = y_min - int(1.3 * text_height)
    txt_y_min = y_min - int(0.3 * text_height)
    if txt_rect_y_min < 0:
        txt_rect_y_min = y_min + int(1.3 * text_height)
        txt_y_min = y_min + int(1 * text_height)

    cv2.rectangle(image, (x_min, txt_rect_y_min), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        image,
        text=txt_label,
        org=(x_min, txt_y_min),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=txt_color,
        lineType=cv2.LINE_AA,
    )
    return image


def plot_yolo_labels(
    image, bboxes_xywhn, class_ids, confidences, id2name=ID2NAME, id2color=ID2COLOR, plot=False
):
    img = image.copy()
    H, W, C = img.shape
    if not isinstance(bboxes_xywhn, np.ndarray):
        bboxes_xywhn = np.array(bboxes_xywhn)
    bboxes_xywh = xywhn2xywh(bboxes_xywhn, W, H)
    bboxes_xyxy = xywh2xyxy(bboxes_xywh).tolist()
    for bbox, class_id, conf in zip(bboxes_xyxy, class_ids, confidences):
        class_name = id2name[class_id]
        color = id2color[class_id]
        img = plot_bbox(img, bbox, class_name, conf, color)
    if plot:
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.imshow(img)
    return img
