# title: U-Net
# zh: U-Net 分割网络
# description: Encoder-decoder from 64 to 1024 channels, skip connections joined by concatenation; padded convolutions keep the input size.
# description_zh: 64 到 1024 通道的编码器-解码器，跳跃连接通过拼接汇合；卷积带填充，输出与输入同尺寸。
# paper: https://arxiv.org/abs/1505.04597
# inputs: x=B,1,128,128
import torch
from torch import nn


# The paper's unpadded convolutions crop the skip connections; with padding=1 the
# sizes match and no cropping is needed (the common modern variant).
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = torch.cat([skip, self.up(x)], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
