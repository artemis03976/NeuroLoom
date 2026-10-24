# title: VAE
# zh: 变分自编码器（MNIST）
# description: Encodes to a 20-dimensional Gaussian latent, samples it by reparameterization and decodes back to 784 pixels.
# description_zh: 编码为 20 维高斯隐变量，用重参数化采样，再解码回 784 个像素。
# paper: https://arxiv.org/abs/1312.6114
# inputs: x=B,784
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Reparameterization: z = mu + sigma * eps with eps ~ N(0, I).
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar
