# title: Transformer block
# zh: Transformer 块
# description: Pre-norm: x + attention(LayerNorm(x)), then x + MLP(LayerNorm(x)), with the expanded multi-head attention.
# description_zh: Pre-LN：x + 注意力(LayerNorm(x))，再 x + MLP(LayerNorm(x))，注意力为展开的多头注意力。
# class: TransformerBlock
# inputs: x=B,T,256
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.k_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.v_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_heads(self.q_proj(x)).transpose(1, 2)
        k = self.k_heads(self.k_proj(x)).transpose(1, 2)
        v = self.v_heads(self.v_proj(x)).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).flatten(2)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))
