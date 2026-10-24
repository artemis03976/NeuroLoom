# title: GPT-2 Small
# zh: GPT-2 Small 语言模型
# description: Decoder-only Transformer: 12 causal self-attention blocks (768 dims, 12 heads), vocabulary 50257, context 1024.
# description_zh: 仅解码器的 Transformer：12 个因果自注意力块（768 维、12 个头），词表 50257，上下文 1024。
# paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# inputs: tokens=B,T:int64
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.k_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.v_heads = nn.Unflatten(-1, (num_heads, self.head_dim))
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_heads(self.q_proj(x)).transpose(1, 2)
        k = self.k_heads(self.k_proj(x)).transpose(1, 2)
        v = self.v_heads(self.v_proj(x)).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask: position i attends only to positions up to i.
        att = att.masked_fill(att.new_ones(att.shape[-2:], dtype=torch.bool).triu(1), float("-inf"))
        att = self.attn_drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).flatten(2)
        return self.resid_drop(self.out_proj(out))


class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, context=1024, dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(context, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, tokens):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.drop(self.wte(tokens) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # The output layer shares its weight with the token embedding (weight tying).
        return x @ self.wte.weight.T
