# title: Squeeze-and-excitation
# zh: SE 通道注意力
# description: Global pooling, two linear layers and a sigmoid give one weight per channel, which rescales the input.
# description_zh: 全局池化、两个线性层和 sigmoid 得到每个通道的权重，再乘回输入。
# inputs: x=B,64,32,32
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels=64, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = torch.flatten(self.pool(x), 1)
        s = self.gate(self.fc2(self.relu(self.fc1(s))))
        return x * s.unsqueeze(-1).unsqueeze(-1)
