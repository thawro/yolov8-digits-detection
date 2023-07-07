from torch import nn
from collections import OrderedDict
from src.yolo.utils import cellboxes_to_boxes, NMS
import torch


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = True,
        maxpool: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
        )
        self.activation = nn.LeakyReLU(0.1)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.maxpool = nn.MaxPool2d(2, 2) if maxpool else None

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        return x


backbone_config = [
    # channels, kernel_size, stride, maxpool
    # 1
    (64, 7, 2, True),
    # 2
    (192, 3, 1, True),
    # 3
    (128, 1, 1, False),
    (256, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, True),
    # 4
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (512, 1, 1, False),
    (1024, 3, 1, True),
    # 5
    (512, 1, 1, False),
    (1024, 3, 1, False),
    (512, 1, 1, False),
    (1024, 3, 1, False),
    (1024, 3, 1, False),
    (1024, 3, 2, False),
    # 6
    (1024, 3, 1, False),
    (1024, 3, 1, False),
]


class YOLOv1Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = 3
        layers = []
        for i, layer_cfg in enumerate(config):
            out_channels, kernel_size, stride, maxpool = layer_cfg
            layers.append(
                (
                    f"layer_{i}",
                    CNNBlock(in_channels, out_channels, kernel_size, stride, True, maxpool),
                )
            )
            in_channels = out_channels
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)

    @property
    def out_channels(self):
        return self.config[-1][0]


class YOLOv1Head(nn.Module):
    def __init__(self, backbone_out_channels: int, S: int, C: int, B: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=backbone_out_channels * S * S, out_features=496),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class YOLOv1(nn.Module):
    def __init__(self, S: int, C: int, B: int):
        self.S = S
        self.C = C
        self.B = B
        super().__init__()
        self.backbone = YOLOv1Backbone(backbone_config)
        self.head = YOLOv1Head(self.backbone.out_channels, S, C, B)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.reshape(-1, self.S, self.S, self.C + self.B * 5)

    def inference(self, x):
        """
        cell_boxes shape: [batch_size, S, S, C + B * 5]
        preds shape: [batch_size, S * S, 6]
        best_class, objectness, x, y, w, h
        """
        cell_boxes = self(x).to("cpu")
        boxes_preds = cellboxes_to_boxes(cell_boxes, self.S, self.C, self.B)
        return boxes_preds

    def perform_nms(
        self,
        boxes_preds: torch.Tensor,
        iou_threshold: float = 0.5,
        objectness_threshold: float = 0.4,
    ):
        all_nms_boxes = []
        for i, boxes in enumerate(boxes_preds):
            nms_boxes = NMS(
                boxes,
                iou_threshold=iou_threshold,
                objectness_threshold=objectness_threshold,
            )
            all_nms_boxes.append(nms_boxes)
        return all_nms_boxes
