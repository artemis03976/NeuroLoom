# title: VGG-style CNN
# zh: VGG 风格卷积网络（CIFAR-10）
# description: Stacked 3×3 convolutions with batch norm, pooling and a dropout classifier.
# description_zh: 堆叠的 3×3 卷积加批归一化与池化，分类头带 Dropout。
# inputs: x=B,3,32,32
from torch import nn


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(self.act4(self.bn4(self.conv4(x))))
        x = self.flatten(self.gap(x))
        return self.fc(self.drop(x))
