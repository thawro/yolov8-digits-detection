from torch import nn


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
