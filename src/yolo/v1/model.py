import torch
from torch import nn
from collections import OrderedDict


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
        if self.maxpool is not None:
            x = self.maxpool(x)
        return self.activation(x)


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
    (1024, 3, 1, False),
    (1024, 3, 2, False),
    # 6
    (1024, 3, 1, False),
    (1024, 3, 1, False),
]


class YOLOv1Backbone(nn.Module):
    def __init__(self, config: list[tuple[int, int, int, bool]]):
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
        self.S = S
        self.C = C
        self.B = B
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=backbone_out_channels * S * S, out_features=496),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

    def forward(self, x):
        out = self.net(x)
        return out.reshape(-1, self.S, self.S, self.C + self.B * 5)


class YOLOv1Model(nn.Module):
    def __init__(self, S: int, C: int, B: int):
        super().__init__()
        self.backbone = YOLOv1Backbone(backbone_config)
        self.head = YOLOv1Head(self.backbone.out_channels, S, C, B)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
