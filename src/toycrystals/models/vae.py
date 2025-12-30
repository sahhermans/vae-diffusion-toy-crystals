from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondVAE(nn.Module):
    def __init__(self, z_dim: int = 16, n_types: int = 4, y_cont_dim: int = 4, cond_drop: float = 0.1) -> None:
        super().__init__()
        self.z_dim = z_dim # latent dim
        self.n_types = n_types # number of categorical types (lattice types)
        self.y_cont_dim = y_cont_dim # number of continuous conditionals
        self.y_dim = n_types + y_cont_dim  # one-hot + continuous
        self.cond_drop = float(cond_drop) # fraction of conditionals to drop during training

        # Encoder: x -> feature map -> vector
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(256 * 4 * 4 + self.y_dim, 256)
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        # Decoder: (z, y) -> feature map -> x
        self.dec_fc = nn.Linear(z_dim + self.y_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def _y_vec(self, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        # y_cat: [B] int64, y_cont: [B,4] float32
        y_oh = F.one_hot(y_cat, num_classes=self.n_types).to(dtype=torch.float32)
        return torch.cat([y_oh, y_cont.to(dtype=torch.float32)], dim=1)

    def encode(self, x: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        y = self._y_vec(y_cat, y_cont)
        h = torch.cat([h, y], dim=1)
        h = F.relu(self.enc_fc(h))
        return self.mu(h), self.logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        y = self._y_vec(y_cat, y_cont)
        
        if self.training and self.cond_drop > 0.0:
            keep = (torch.rand((y.shape[0], 1), device=y.device) >= self.cond_drop).to(y.dtype)
            y = y * keep

        h = self.dec_fc(torch.cat([z, y], dim=1)).view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(
        self, x: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y_cat, y_cont)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z, y_cat, y_cont)
        return x_hat, mu, logvar


class VAE(nn.Module):
    """Unconditional VAE baseline (no y_cat / y_cont anywhere)."""

    def __init__(self, z_dim: int = 16) -> None:
        super().__init__()
        self.z_dim = z_dim

        # Encoder: x -> feature map -> vector
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(256 * 4 * 4, 256)
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        # Decoder: z -> feature map -> x
        self.dec_fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        h = F.relu(self.enc_fc(h))
        return self.mu(h), self.logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar