import torch
from torch import nn
from src.yolo.utils import calculate_boxes_iou


class YoloLoss(nn.Module):
    def __init__(
        self, S: int, C: int, B: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5
    ):
        super().__init__()
        self.C = C
        self.B = B
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def _box_loss(
        self, obj_preds: torch.Tensor, obj_targets: torch.Tensor, bestbox_xywh_idxs: torch.Tensor
    ):
        """Loss of box coordinates (x, y, w, h).
        The loss is calculated only for cells with objects present.

        Args:
            obj_preds (torch.Tensor): Predicted boxes for cells with objects.
            obj_targets (torch.Tensor): Target boxes for cells with objects.
            bestbox_xywh_idxs (torch.Tensor): Indices of best boxes xywh coords. For each cell there are B boxes predicted
                and only the best of those is considered.

        Returns:
            float: Box loss.
        """
        # N x 4
        # 4 for: [x, y, w, h]
        obj_bestbox_preds = obj_preds.gather(1, bestbox_xywh_idxs)
        obj_box_targets = obj_targets[:, self.C + 1 : self.C + 5]

        obj_bestbox_preds_xy = obj_bestbox_preds[:, :2]
        obj_box_targets_xy = obj_box_targets[:, :2]

        obj_bestbox_preds_wh = obj_bestbox_preds[:, 2:]
        obj_bestbox_preds_wh = torch.sign(obj_bestbox_preds_wh) * torch.sqrt(
            abs(obj_bestbox_preds_wh) + 1e-6
        )
        obj_box_targets_wh = torch.sqrt(obj_box_targets[:, 2:])

        xy_loss = self.mse(obj_bestbox_preds_xy, obj_box_targets_xy)
        wh_loss = self.mse(obj_bestbox_preds_wh, obj_box_targets_wh)
        return xy_loss + wh_loss

    def _object_loss(
        self, obj_preds: torch.Tensor, obj_targets: torch.Tensor, bestbox_obj_idxs: torch.Tensor
    ):
        """Objectness loss for cells with objects present.

        .. note:: object_loss is calculated for the best box only

        Args:
            obj_preds (torch.Tensor): Predicted boxes for cells with objects.
            obj_targets (torch.Tensor): Target boxes for cells with objects.
            bestbox_obj_idxs (torch.Tensor): Indices of best boxes objectness scores. For each cell there are B boxes predicted
                and only the best of those is considered.

        Returns:
            float: Objectness loss for cells with objects
        """
        obj_object_preds = obj_preds.gather(1, bestbox_obj_idxs).flatten()
        obj_object_targets = obj_targets[:, self.C]
        object_loss = self.mse(obj_object_preds, obj_object_targets)
        return object_loss

    def _no_object_loss(self, noobj_preds: torch.Tensor, noobj_targets: torch.Tensor):
        """Objectness loss for cells without objects present.

        .. note:: no_object_loss is calculated for all boxes (not only the best one)

        Args:
            noobj_preds (torch.Tensor): Predicted boxes for cells without objects.
            noobj_targets (torch.Tensor): Target boxes for cells without objects.

        Returns:
            float: Objectness loss for cells without objects
        """
        no_obj_object_preds = noobj_preds[:, self.C :][:, ::5]
        no_obj_object_targets = noobj_targets[:, self.C :][:, ::5]
        no_object_loss = self.mse(no_obj_object_preds, no_obj_object_targets)
        return no_object_loss

    def _class_loss(self, obj_preds: torch.Tensor, obj_targets: torch.Tensor):
        """Class loss for cells with objects present.

        Args:
            obj_preds (torch.Tensor): Predicted boxes for cells with objects.
            obj_targets (torch.Tensor): Target boxes for cells with objects.

        Returns:
            float: Class loss for cells with objects
        """
        obj_class_preds = obj_preds[:, : self.C]
        obj_class_targets = obj_targets[:, : self.C]
        class_loss = self.mse(obj_class_preds, obj_class_targets)
        return class_loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds and targets of shapes: [batch_size, S, S, C + B * 5]
        boxes_iou = []
        for box_idx in range(self.B):
            box_start_idx = self.C + 1 + box_idx * 5
            box_end_idx = box_start_idx + 5
            box_iou = calculate_boxes_iou(
                preds[..., box_start_idx:box_end_idx], targets[..., self.C + 1 : self.C + 5]
            ).squeeze(-1)
            boxes_iou.append(box_iou.unsqueeze(0))

        ious = torch.cat(boxes_iou, dim=0)

        iou_maxes, bestbox_idxs = torch.max(ious, dim=0)
        obj_mask = targets[..., self.C] == 1  # Iobj_i

        bestbox_idxs = bestbox_idxs[obj_mask]  # only idxs with objects
        obj_preds = preds[obj_mask]  # only preds with objects
        obj_targets = targets[obj_mask]  # only targets with objects

        noobj_preds = preds[~obj_mask]  # only preds without objects
        noobj_targets = targets[~obj_mask]  # only targets without objects

        DEVICE = obj_preds.device
        bestbox_xywh_idxs = torch.tensor(
            [range(box_idx * 5 + self.C + 1, box_idx * 5 + self.C + 5) for box_idx in bestbox_idxs]
        ).to(DEVICE)
        bestbox_obj_idxs = torch.tensor(
            [range(box_idx * 5 + self.C, box_idx * 5 + self.C + 1) for box_idx in bestbox_idxs]
        ).to(DEVICE)

        box_loss = self._box_loss(obj_preds, obj_targets, bestbox_xywh_idxs)
        object_loss = self._object_loss(obj_preds, obj_targets, bestbox_obj_idxs)
        no_object_loss = self._no_object_loss(noobj_preds, noobj_targets)
        class_loss = self._class_loss(obj_preds, obj_targets)
        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
