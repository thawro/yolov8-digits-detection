import numpy as np
import cv2


def non_maximum_supression(boxes_xyxy: np.ndarray, conf: np.ndarray, iou_threshold: float):
    """Apply Non Maximum Supression and return indices of boxes to keep"""
    # Sort by score
    sorted_indices = np.argsort(conf)[::-1]

    keep_boxes_idxs = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0].item()
        keep_boxes_idxs.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = calculate_iou(boxes_xyxy[box_id, :], boxes_xyxy[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious <= iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes_idxs


def calculate_iou(box_xyxy: np.ndarray, boxes_xyxy: np.ndarray):
    """Return Intersection over Union (IoU) between box and other boxes"""
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box_xyxy[0], boxes_xyxy[:, 0])
    ymin = np.maximum(box_xyxy[1], boxes_xyxy[:, 1])
    xmax = np.minimum(box_xyxy[2], boxes_xyxy[:, 2])
    ymax = np.minimum(box_xyxy[3], boxes_xyxy[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    union_area = box_area + boxes_area - intersection_area
    return intersection_area / union_area


def xywh2xyxy(boxes_xywh: np.ndarray):
    """Parse boxes format from xywh to xyxy"""
    x, y, w, h = boxes_xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.column_stack((x1, y1, x2, y2))
    return boxes_xyxy.astype(boxes_xywh.dtype)


def xywhn2xywh(boxes_xywhn: np.ndarray, h: int, w: int):
    """Parse boxes format from xywhn to xywh using image height (`h`) and width (`w`)"""
    xn, yn, wn, hn = boxes_xywhn.T
    x = xn * w
    y = yn * h
    w = wn * w
    h = hn * h
    boxes_xywh = np.column_stack((x, y, w, h))
    return boxes_xywh.astype(np.int16)


def resize_pad(
    image: np.ndarray, h: int = 640, w: int = 640, fill_value: int = 114
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize and Pad an image to reach desired output size (`h x w`).

    Args:
        image (np.ndarray): Image to process.
        h (int, optional): Desired height. Defaults to 640.
        w (int, optional): Desired width. Defaults to 640.
        fill_value (int, optional): Scalar value used for padding. Defaults to 114.

    Returns:
        tuple[np.ndarray, tuple[int, int, int, int]]: (image, padding_tlbr) tuple with
            processed image and padding values.
            Padding values order: [top, left, bottom, right]
    """
    img_h, img_w, img_c = image.shape
    aspect_ratio = img_w / img_h
    if aspect_ratio > 1:
        new_img_w = w
        new_img_h = int(w / aspect_ratio)
    else:
        new_img_h = h
        new_img_w = int(h * aspect_ratio)
    resized_img = cv2.resize(image, (new_img_w, new_img_h))
    print(resized_img.shape)
    # width, height ratios
    padded_img = np.ones((h, w, img_c)) * fill_value
    pad_x = w - new_img_w
    pad_y = h - new_img_h

    pad_top = pad_y // 2
    pad_left = pad_x // 2
    pad_bottom = pad_y - pad_top
    pad_right = pad_x - pad_left
    padded_img[pad_bottom : pad_bottom + new_img_h, pad_left : pad_left + new_img_w] = resized_img
    return padded_img, (pad_top, pad_left, pad_bottom, pad_right)
