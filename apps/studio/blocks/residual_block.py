# title: Residual block
# zh: 残差块
# description: Two 3×3 convolutions with batch norm plus a 1×1 projection shortcut; for an identity shortcut, delete it and connect x to the add.
# description_zh: 两个带批归一化的 3×3 卷积加 1×1 投影捷径；需要恒等捷径时删除捷径，把 x 直接连到加法。
# inputs: x=B,64,56,56
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + self.shortcut(x))
