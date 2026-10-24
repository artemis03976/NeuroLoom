# title: U-Net down block
# zh: U-Net 下采样块
# description: 2×2 max pooling halves the resolution, then two 3×3 convolutions with ReLU.
# description_zh: 2×2 最大池化把分辨率减半，再接两个带 ReLU 的 3×3 卷积。
# class: Down
# inputs: x=B,64,64,64
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


class Down(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))
