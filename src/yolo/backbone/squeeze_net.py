"""SqueezeNet architecture based on https://arxiv.org/pdf/1602.07360.pdf.
By default, the simple bypass version is used
Also, BatchNormalization is added for each squeeze and expand layers"""

import torch
from torch import nn
from collections import OrderedDict
from typing import Literal


class Backbone(nn.Module):
    def __init__(self, net: nn.Module, out_channels: int, name: str):
        super().__init__()
        self.net = net
        self.out_channels = out_channels
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class CNNBlock(nn.Module):
    """Single CNN block constructed of combination of Conv2d, Activation, Pooling, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: str | int | tuple = 0,
        groups: int = 1,
        activation: str | None = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of Conv2d input channels.
            out_channels (int): Number of Conv2d out channels.
            kernel_size (int): Conv2d kernel equal to `(kernel_size, kernel_size)`.
            stride (int, optional): Conv2d stride equal to `(stride, stride)`.
                Defaults to 1.
            padding (int | str, optional): Conv2d padding equal to `(padding, padding)`.
                Defaults to 1.. Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_fn = nn.LeakyReLU(0.1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation_fn(out)
        return out


class FireBlock(nn.Module):
    """FireBlock used to squeeze and expand convolutional channels"""

    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float,
        expand_filters: int,
        pct_3x3: float,
        is_residual: bool = False,
    ):
        super().__init__()
        s_1x1 = int(squeeze_ratio * expand_filters)
        e_3x3 = int(expand_filters * pct_3x3)
        e_1x1 = expand_filters - e_3x3
        self.squeeze_1x1 = CNNBlock(in_channels, s_1x1, kernel_size=1)
        self.expand_1x1 = CNNBlock(s_1x1, e_1x1, kernel_size=1)
        self.expand_3x3 = CNNBlock(s_1x1, e_3x3, kernel_size=3, padding=1)
        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_out = self.squeeze_1x1(x)
        expand_1x1_out = self.expand_1x1(squeeze_out)
        expand_3x3_out = self.expand_3x3(squeeze_out)
        out = torch.concat([expand_1x1_out, expand_3x3_out], dim=1)  # concat over channels
        if self.is_residual:
            return x + out
        return out


class SqueezeNet1_0(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_e: int = 128,
        incr_e: int = 128,
        pct_3x3: float = 0.5,
        freq: int = 2,
        SR: float = 0.125,
        simple_bypass: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_e = base_e
        self.incr_e = incr_e
        self.pct_3x3 = pct_3x3
        self.freq = freq
        self.SR = SR

        # architecture, fb - fire block
        out_channels = 96
        n_fire_blocks = 8
        fb_expand_filters = [base_e + (incr_e * (i // freq)) for i in range(n_fire_blocks)]
        fb_in_channels = [out_channels] + fb_expand_filters
        is_residual = [False] + [(i % freq == 1 and simple_bypass) for i in range(1, n_fire_blocks)]
        self.fb_in_channels = fb_in_channels
        self.out_channels = fb_expand_filters[-1]
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2)
        maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire2 = FireBlock(fb_in_channels[0], SR, fb_expand_filters[0], pct_3x3, is_residual[0])
        fire3 = FireBlock(fb_in_channels[1], SR, fb_expand_filters[1], pct_3x3, is_residual[1])
        fire4 = FireBlock(fb_in_channels[2], SR, fb_expand_filters[2], pct_3x3, is_residual[2])
        maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire5 = FireBlock(fb_in_channels[3], SR, fb_expand_filters[3], pct_3x3, is_residual[3])
        fire6 = FireBlock(fb_in_channels[4], SR, fb_expand_filters[4], pct_3x3, is_residual[4])
        fire7 = FireBlock(fb_in_channels[5], SR, fb_expand_filters[5], pct_3x3, is_residual[5])
        fire8 = FireBlock(fb_in_channels[6], SR, fb_expand_filters[6], pct_3x3, is_residual[6])
        maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire9 = FireBlock(fb_in_channels[7], SR, fb_expand_filters[7], pct_3x3, is_residual[7])
        dropout9 = nn.Dropout2d(p=0.5)
        layers = [
            ("conv1", conv1),
            ("maxpool1", maxpool1),
            ("fire2", fire2),
            ("fire3", fire3),
            ("fire4", fire4),
            ("maxpool4", maxpool4),
            ("fire5", fire5),
            ("fire6", fire6),
            ("fire7", fire7),
            ("fire8", fire8),
            ("maxpool8", maxpool8),
            ("fire9", fire9),
            ("dropout9", dropout9),
        ]
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def name(self):
        return "SqueezeNet"


class SqueezeNetBackbone:
    def __new__(
        cls,
        in_channels: int,
        version: Literal["squeezenet1_0", "squeezenet1_1"],
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        if load_from_torch:
            _net = torch.hub.load("pytorch/vision:v0.10.0", version, pretrained=pretrained)
            _net.classifier = torch.nn.Identity()
            _net = _net.features

        else:
            _net = SqueezeNet1_0(in_channels=in_channels).net
        net = Backbone(
            nn.Sequential(
                _net,
                nn.Conv2d(512, 256, 3, 2, 1),
                nn.Conv2d(256, 512, 1, 1, 0),
                nn.Conv2d(512, 256, 3, 2, 1),  # out is 7x7
            ),
            out_channels=256,
            name=version,
        )
        if freeze_extractor:
            net.freeze()
        return net
