# title: Bottleneck block
# zh: 瓶颈残差块
# description: 1×1, 3×3 and 1×1 convolutions (4× expansion) with batch norm plus a 1×1 projection shortcut, as in ResNet-50.
# description_zh: 1×1、3×3、1×1 卷积（通道扩张 4 倍）带批归一化，加 1×1 投影捷径，同 ResNet-50。
# inputs: x=B,64,56,56
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels=64, width=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, width * 4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(width * 4),
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu3(out + self.shortcut(x))
