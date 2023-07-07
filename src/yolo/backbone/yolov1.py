from torch import nn
from collections import OrderedDict
from src.yolo.helpers import CNNBlock


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
    def __init__(self, config: list[tuple[int, int, int, bool]] = backbone_config):
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
