# title: Tiny U-Net
# zh: 迷你 U-Net 分割网络
# description: An encoder-decoder with skip connections joined by concatenation.
# description_zh: 编码器-解码器结构，跳跃连接通过拼接汇合。
# inputs: x=B,3,64,64
import torch
from torch import nn


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.mid = nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = nn.ReLU()
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.act4 = nn.ReLU()
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.act5 = nn.ReLU()
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s1 = self.act1(self.enc1(x))
        s2 = self.act2(self.enc2(self.pool1(s1)))
        m = self.act3(self.mid(self.pool2(s2)))
        d2 = self.act4(self.dec2(torch.cat([self.up2(m), s2], dim=1)))
        d1 = self.act5(self.dec1(torch.cat([self.up1(d2), s1], dim=1)))
        return self.out(d1)
