# title: ResNet-18
# zh: ResNet-18 残差网络
# description: Four stages of two basic residual blocks (two 3×3 convolutions and a shortcut), ImageNet 224×224.
# description_zh: 四段、每段两个基本残差块（两个 3×3 卷积加捷径），ImageNet 224×224。
# paper: https://arxiv.org/abs/1512.03385
# inputs: x=B,3,224,224
import torch
from torch import nn


# Layer names follow torchvision's resnet18, so its pretrained state_dict loads as is.
# NeuroLoom builds a module's structure from its code without running it, so the
# identity and projection shortcuts are two classes rather than an `if` in __init__.
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + x)


class BasicBlockProj(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + self.downsample(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.layer2 = nn.Sequential(BasicBlockProj(64, 128, 2), BasicBlock(128))
        self.layer3 = nn.Sequential(BasicBlockProj(128, 256, 2), BasicBlock(256))
        self.layer4 = nn.Sequential(BasicBlockProj(256, 512, 2), BasicBlock(512))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)
