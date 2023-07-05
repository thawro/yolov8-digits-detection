import torch


def calculate_boxes_iou(boxes_preds, boxes_targets, format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_targets (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if format == "midpoint":
        boxes_preds_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        boxes_preds_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        boxes_preds_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        boxes_preds_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        boxes_targets_x1 = boxes_targets[..., 0:1] - boxes_targets[..., 2:3] / 2
        boxes_targets_y1 = boxes_targets[..., 1:2] - boxes_targets[..., 3:4] / 2
        boxes_targets_x2 = boxes_targets[..., 0:1] + boxes_targets[..., 2:3] / 2
        boxes_targets_y2 = boxes_targets[..., 1:2] + boxes_targets[..., 3:4] / 2

    elif format == "corners":
        boxes_preds_x1 = boxes_preds[..., 0:1]
        boxes_preds_y1 = boxes_preds[..., 1:2]
        boxes_preds_x2 = boxes_preds[..., 2:3]
        boxes_preds_y2 = boxes_preds[..., 3:4]  # (N, 1)
        boxes_targets_x1 = boxes_targets[..., 0:1]
        boxes_targets_y1 = boxes_targets[..., 1:2]
        boxes_targets_x2 = boxes_targets[..., 2:3]
        boxes_targets_y2 = boxes_targets[..., 3:4]
    else:
        raise ValueError("Wrong format passed")

    x1 = torch.max(boxes_preds_x1, boxes_targets_x1)
    y1 = torch.max(boxes_preds_y1, boxes_targets_y1)
    x2 = torch.min(boxes_preds_x2, boxes_targets_x2)
    y2 = torch.min(boxes_preds_y2, boxes_targets_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    boxes_preds_area = abs((boxes_preds_x2 - boxes_preds_x1) * (boxes_preds_y2 - boxes_preds_y1))
    boxes_targets_area = abs(
        (boxes_targets_x2 - boxes_targets_x1) * (boxes_targets_y2 - boxes_targets_y1)
    )

    return intersection / (boxes_preds_area + boxes_targets_area - intersection + 1e-6)
