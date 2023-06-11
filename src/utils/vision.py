import numpy as np


def non_maximum_supression(boxes_xyxy: np.ndarray, conf: np.ndarray, iou_threshold: float):
    # Sort by score
    sorted_indices = np.argsort(conf)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0].item()
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = calculate_iou(boxes_xyxy[box_id, :], boxes_xyxy[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious <= iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def calculate_iou(box_xyxy: np.ndarray, boxes_xyxy: np.ndarray):
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
