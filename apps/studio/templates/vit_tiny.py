# title: ViT-Tiny
# zh: ViT-Tiny 视觉 Transformer
# description: 16×16 patches, a class token and position embeddings, then 12 Transformer blocks (192 dims, 3 heads).
# description_zh: 切成 16×16 图块，加类别标记与位置编码，再经过 12 个 Transformer 块（192 维、3 个头）。
# paper: https://arxiv.org/abs/2010.11929
# inputs: x=B,3,224,224
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=192, depth=12, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        # Prepend the class token and add the position embeddings.
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]  # classify from the class token
        return self.head(cls)
