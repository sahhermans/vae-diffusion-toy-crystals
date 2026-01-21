from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    t: [B] int64
    returns: [B, dim] float32
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0, math.log(10_000), steps=half, device=t.device, dtype=torch.float32) * (-1.0)
    )  # [half]
    args = t.to(torch.float32)[:, None] * freqs[None, :]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=1)
    return emb


def y_vec(y_cat: torch.Tensor, y_cont: torch.Tensor, n_types: int) -> torch.Tensor:
    """
    Matches CondVAE._y_vec: one-hot(y_cat) concatenated with y_cont. 
    y_cat: [B] int64, 
    y_cont: [B, y_cont_dim] float32
    returns: [B, n_types + y_cont_dim] float32
    """
    y_oh = F.one_hot(y_cat, num_classes=n_types).to(dtype=torch.float32)
    return torch.cat([y_oh, y_cont.to(dtype=torch.float32)], dim=1)


class FiLMResBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int, mult: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, mult * width)
        self.fc2 = nn.Linear(mult * width, width)
        self.cond = nn.Linear(cond_dim, 2 * width)  # gamma, beta
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, width], cond: [B, cond_dim]
        h = self.norm(x)
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


class DiffusionPriorFiLM(nn.Module):
    """
    Drop-in replacement for DiffusionPrior:
    predicts epsilon given z_t, t, y_cat, y_cont.
    Uses residual MLP blocks with FiLM conditioning from (t,y).
    """
    def __init__(
        self,
        z_dim: int,
        n_types: int,
        y_cont_dim: int,
        t_emb_dim: int = 64,
        width: int = 256,
        n_blocks: int = 6,
        y_cat_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_types = int(n_types)
        self.y_cont_dim = int(y_cont_dim)
        self.t_emb_dim = int(t_emb_dim)

        # Categorical conditioning: learn an embedding instead of one-hot
        self.y_cat_emb = nn.Embedding(self.n_types, y_cat_emb_dim)

        # Continuous conditioning (optionally more expressive than raw concat)
        self.y_cont_mlp = nn.Sequential(
            nn.Linear(self.y_cont_dim, y_cat_emb_dim),
            nn.SiLU(),
            nn.Linear(y_cat_emb_dim, y_cat_emb_dim),
        )

        # Fuse y features into width
        self.y_fuse = nn.Sequential(
            nn.Linear(2 * y_cat_emb_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

        # Project timestep embedding into width
        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_emb_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

        self.in_proj = nn.Linear(self.z_dim, width)

        # Conditioning vector is concat(t_feat, y_feat)
        cond_dim = 2 * width
        self.blocks = nn.ModuleList([FiLMResBlock(width, cond_dim) for _ in range(n_blocks)])

        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, self.z_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        # t: [B] int64, y_cat: [B] int64, y_cont: [B, y_cont_dim]
        te = timestep_embedding(t, self.t_emb_dim)          # [B, t_emb_dim]
        t_feat = self.t_mlp(te)                              # [B, width]

        y_cat_feat = self.y_cat_emb(y_cat)                   # [B, y_cat_emb_dim]
        y_cont_feat = self.y_cont_mlp(y_cont.to(torch.float32))  # [B, y_cat_emb_dim]
        y_feat = self.y_fuse(torch.cat([y_cat_feat, y_cont_feat], dim=-1))  # [B, width]

        cond = torch.cat([t_feat, y_feat], dim=-1)           # [B, 2*width]

        h = self.in_proj(z_t)                                # [B, width]
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out_proj(self.out_norm(h))                  # [B, z_dim]
        return h


class DiffusionPrior(nn.Module):
    """
    Predicts epsilon given z_t, t, and y.
    """
    def __init__(self, z_dim: int, n_types: int, y_cont_dim: int, t_emb_dim: int = 64, width: int = 256) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_types = int(n_types)
        self.y_cont_dim = int(y_cont_dim)
        self.t_emb_dim = int(t_emb_dim)

        in_dim = self.z_dim + (self.n_types + self.y_cont_dim) + self.t_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, self.z_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B, z_dim]
        t: [B] int64
        y_cat: [B] int64
        y_cont: [B, y_cont_dim] float32
        returns: eps_pred [B, z_dim]
        """
        te = timestep_embedding(t, self.t_emb_dim)              # [B, t_emb_dim]
        y = y_vec(y_cat, y_cont, n_types=self.n_types)          # [B, n_types + y_cont_dim]
        h = torch.cat([z_t, y, te], dim=1)
        return self.net(h)


@dataclass(frozen=True)
class DiffusionSchedule:
    """
    Precomputes DDPM constants for a linear beta schedule.
    """
    betas: torch.Tensor          # [T]
    alphas: torch.Tensor         # [T]
    alpha_bars: torch.Tensor     # [T]
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @staticmethod
    def linear(T: int, beta_start: float, beta_end: float, device: torch.device) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, steps=T, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return DiffusionSchedule(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=torch.sqrt(alpha_bars),
            sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        )

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: z_t = sqrt(abar_t) * z0 + sqrt(1-abar_t) * eps
        z0: [B, z_dim], t: [B] int64, eps: [B, z_dim]
        """
        a = self.sqrt_alpha_bars[t].unsqueeze(1)               # [B,1]
        b = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)     # [B,1]
        return a * z0 + b * eps

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        y_cat: torch.Tensor,
        y_cont: torch.Tensor,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (eta=0 -> deterministic).
        returns z0: [B, z_dim]
        """
        model.eval()
        device = self.betas.device
        B = int(y_cat.shape[0])
        z = torch.randn((B, model.z_dim), device=device)

        # Pick a subset of timesteps (descending).
        T = int(self.betas.shape[0])
        ts = torch.linspace(T - 1, 0, steps=n_steps, device=device)
        ts = torch.round(ts).to(torch.int64)
        ts = torch.unique_consecutive(ts)

        n = int(ts.numel())
        for i in range(n):
            t = ts[i].repeat(B)  # [B]
            eps_pred = model(z, t, y_cat, y_cont)  # [B, z_dim]

            abar_t = self.alpha_bars[t].unsqueeze(1)  # [B,1]
            sqrt_abar_t = torch.sqrt(abar_t)
            sqrt_1m_abar_t = torch.sqrt(1.0 - abar_t)

            # Predict z0 from current z_t
            z0_pred = (z - sqrt_1m_abar_t * eps_pred) / (sqrt_abar_t + 1e-8)

            if i == n - 1:
                z = z0_pred
                break

            t_prev = ts[i + 1].repeat(B)
            abar_prev = self.alpha_bars[t_prev].unsqueeze(1)
            sqrt_abar_prev = torch.sqrt(abar_prev)
            sqrt_1m_abar_prev = torch.sqrt(1.0 - abar_prev)

            # eta=0 DDIM update (no extra noise term)
            if eta != 0.0:
                # Optional: stochasticity; kept simple.
                raise NotImplementedError("eta != 0 not implemented in this minimal version")

            z = sqrt_abar_prev * z0_pred + sqrt_1m_abar_prev * eps_pred

        return z
