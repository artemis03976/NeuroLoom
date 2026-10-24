# title: LSTM text classifier
# zh: LSTM 文本分类
# description: Token embeddings, a bidirectional LSTM, mean pooling over time and a linear head.
# description_zh: 词嵌入、双向 LSTM、按时间步平均池化，再接线性分类头。
# paper: https://doi.org/10.1162/neco.1997.9.8.1735
# inputs: tokens=B,T:int64
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 4)

    def forward(self, tokens):
        out, (h, c) = self.lstm(self.embed(tokens))
        return self.fc(self.drop(out.mean(1)))
