# title: Transformer encoder classifier
# zh: Transformer 编码器分类
# description: Token embeddings, two pre-norm Transformer encoder layers, mean pooling and a linear head.
# description_zh: 词嵌入、两层 Pre-LN Transformer 编码层、平均池化，再接线性分类头。
# inputs: tokens=B,T:int64
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 256)
        self.encoder1 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=1024, activation="gelu", batch_first=True, norm_first=True)
        self.encoder2 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=1024, activation="gelu", batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 2)

    def forward(self, tokens):
        x = self.encoder2(self.encoder1(self.embed(tokens)))
        return self.head(self.norm(x.mean(1)))
