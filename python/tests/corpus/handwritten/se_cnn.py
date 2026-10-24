# style: squeeze-and-excitation blocks using runtime sizes, nn.Sequential heads
# inputs: x=B,3,32,32
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.se = SELayer(out_ch)

    def forward(self, x):
        return self.se(self.act(self.bn(self.conv(x))))


class SENet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = SEConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = SEConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, num_classes))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        return self.classifier(x)
