# title: MLP
# zh: 多层感知机（MNIST）
# description: Two hidden layers with ReLU and dropout, for 28×28 digits flattened to 784.
# description_zh: 两个隐藏层，带 ReLU 与 Dropout，输入为展平的 28×28 手写数字。
# inputs: x=B,784
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
