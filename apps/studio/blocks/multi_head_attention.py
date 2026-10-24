# title: Multi-head attention
# zh: 多头注意力
# description: Q, K and V projections split into heads, scaled dot-product softmax, heads merged and an output projection.
# description_zh: Q、K、V 投影拆成多头，缩放点积后 softmax，合并各头再接输出投影。
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
