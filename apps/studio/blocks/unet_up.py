# title: U-Net up block
# zh: U-Net 上采样块
# description: A 2×2 transposed convolution doubles the resolution, the skip connection is concatenated, then two 3×3 convolutions.
# description_zh: 2×2 转置卷积把分辨率加倍，与跳跃连接拼接，再接两个 3×3 卷积。
# class: Up
# inputs: x=B,128,32,32; skip=B,64,64,64
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


class Up(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = torch.cat([skip, self.up(x)], dim=1)
        return self.conv(x)
