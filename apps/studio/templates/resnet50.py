# title: ResNet-50
# zh: ResNet-50 残差网络
# description: Four stages of 3, 4, 6 and 3 bottleneck blocks (1×1, 3×3 and 1×1 convolutions), ImageNet 224×224.
# description_zh: 四段瓶颈残差块，依次 3、4、6、3 个（1×1、3×3、1×1 卷积），ImageNet 224×224。
# paper: https://arxiv.org/abs/1512.03385
# inputs: x=B,3,224,224
import torch
from torch import nn


# Layer names follow torchvision's resnet50 (stride on the 3×3 convolution, "v1.5"),
# so its pretrained state_dict loads as is. NeuroLoom builds a module's structure
# from its code without running it, so the identity and projection shortcuts are
# two classes rather than an `if` in __init__.
class Bottleneck(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width * 4, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu3(out + x)


class BottleneckProj(nn.Module):
    def __init__(self, in_channels, width, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.relu3 = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, width * 4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(width * 4),
        )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu3(out + self.downsample(x))


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BottleneckProj(64, 64, 1),
            Bottleneck(64),
            Bottleneck(64),
        )
        self.layer2 = nn.Sequential(
            BottleneckProj(256, 128, 2),
            Bottleneck(128),
            Bottleneck(128),
            Bottleneck(128),
        )
        self.layer3 = nn.Sequential(
            BottleneckProj(512, 256, 2),
            Bottleneck(256),
            Bottleneck(256),
            Bottleneck(256),
            Bottleneck(256),
            Bottleneck(256),
        )
        self.layer4 = nn.Sequential(
            BottleneckProj(1024, 512, 2),
            Bottleneck(512),
            Bottleneck(512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)
