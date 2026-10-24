# title: VGG-11
# zh: VGG-11 卷积网络
# description: Eight 3×3 convolutions in five pooled stages, then three fully connected layers (ImageNet, 224×224).
# description_zh: 八个 3×3 卷积分五段、每段后接池化，再接三个全连接层（ImageNet，224×224）。
# paper: https://arxiv.org/abs/1409.1556
# inputs: x=B,3,224,224
import torch
from torch import nn


# Configuration "A" of the paper. Layer names follow torchvision's vgg11, so its
# pretrained state_dict loads as is.
class VGG11(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        x = torch.flatten(x, 1)
        return self.classifier(x)
