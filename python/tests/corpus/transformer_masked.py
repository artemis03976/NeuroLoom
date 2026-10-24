# style: text classifier on nn.TransformerEncoder with an optional padding mask argument
# inputs: tokens=B,10:int64
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    def __init__(self, vocab=200, d_model=32, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens, padding_mask=None):
        x = self.pos(self.embed(tokens) * math.sqrt(self.embed.embedding_dim))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(x.mean(dim=1))
