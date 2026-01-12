from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Embeddings
# =========================

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Continuous-time sinusoidal embedding.

    t: [B] in [0,1]
    returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )  # [half]
    args = (2.0 * math.pi) * t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionEmbedding(nn.Module):
    """
    Embed (y_cat, y_cont) into a single conditioning vector.

    Committed choice:
      - theta (assumed at y_cont[:, 1]) is always encoded as sin/cos:
            y_cont[:, 1] = sin(theta)
            y_cont[:, 2] = cos(theta)
        This matches the rot_only dataset where y_cont[2] is unused (0.0).

    CFG:
      - Reserve y_cat == n_types as the "null" token.
    """

    def __init__(self, n_types: int, y_cont_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.n_types = int(n_types)
        self.y_cont_dim = int(y_cont_dim)
        self.emb_dim = int(emb_dim)

        if self.y_cont_dim < 3:
            raise ValueError("theta_sincos requires y_cont_dim >= 3 (needs indices 1 and 2).")

        self.cat_emb = nn.Embedding(self.n_types + 1, emb_dim)  # +1 for null token (CFG)
        self.cont_mlp = nn.Sequential(
            nn.Linear(self.y_cont_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        y_cat = y_cat.clamp(min=0, max=self.n_types).to(torch.long)
        y_cont = y_cont.float()

        # Always encode theta as sin/cos in-place.
        # For rot_only: y_cont = [0, theta, 0, 0] -> [0, sin(theta), cos(theta), 0]
        y = y_cont.clone()
        theta = y[:, 1]
        y[:, 1] = torch.sin(theta)
        y[:, 2] = torch.cos(theta)

        e_cat = self.cat_emb(y_cat)  # [B, emb_dim]
        e_cont = self.cont_mlp(y)    # [B, emb_dim]
        return self.out(torch.cat([e_cat, e_cont], dim=1))


# =========================
# Tiny conditional U-Net
# =========================

def _gn_groups(ch: int) -> int:
    # Small helper to pick a valid number of groups.
    for g in (8, 4, 2):
        if ch % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        g = _gn_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CondUNetTiny(nn.Module):
    """
    Tiny U-Net predicting eps_hat = εθ(x_t, t, c).

    Improvements committed:
      (2) circular padding everywhere
      (3) GroupNorm in conv blocks
      (5) upsample via bilinear + conv
    """

    def __init__(
        self,
        n_types: int,
        y_cont_dim: int,
        base_ch: int = 32,
        emb_dim: int = 128,
        cond_ch: int = 8,
        time_ch: int = 8,
    ) -> None:
        super().__init__()

        self.n_types = int(n_types)
        self.y_cont_dim = int(y_cont_dim)

        self.cond_emb = ConditionEmbedding(n_types=self.n_types, y_cont_dim=self.y_cont_dim, emb_dim=emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.to_cond_map = nn.Linear(emb_dim, cond_ch)
        self.to_time_map = nn.Linear(emb_dim, time_ch)

        in_ch = 1 + cond_ch + time_ch

        # Down
        self.down1 = _ConvBlock(in_ch, base_ch)
        self.ds1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1, padding_mode="circular")
        self.down2 = _ConvBlock(base_ch, base_ch * 2)
        self.ds2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1, padding_mode="circular")

        # Bottleneck
        self.mid = _ConvBlock(base_ch * 2, base_ch * 2)

        # Up (nearest + conv to reduce checkerboards)
        self.us2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.us2_conv = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, padding_mode="circular")
        self.up2 = _ConvBlock(base_ch * 4, base_ch)

        self.us1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.us1_conv = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, padding_mode="circular")
        self.up1 = _ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1, padding_mode="circular")

    def _make_maps(
        self,
        t: torch.Tensor,
        y_cat: torch.Tensor,
        y_cont: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.cond_emb.emb_dim)  # [B, emb_dim]
        t_emb = self.time_mlp(t_emb)                          # [B, emb_dim]
        c_emb = self.cond_emb(y_cat, y_cont)                  # [B, emb_dim]

        t_map = self.to_time_map(t_emb).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        c_map = self.to_cond_map(c_emb).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return torch.cat([t_map, c_map], dim=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y_cat: torch.Tensor, y_cont: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_t.shape
        maps = self._make_maps(t, y_cat, y_cont, H, W)
        x = torch.cat([x_t, maps], dim=1)

        h1 = self.down1(x)   # [B, base_ch, H, W]
        h = self.ds1(h1)     # [B, base_ch, H/2, W/2]
        h2 = self.down2(h)   # [B, 2*base_ch, H/2, W/2]
        h = self.ds2(h2)     # [B, 2*base_ch, H/4, W/4]

        h = self.mid(h)      # [B, 2*base_ch, H/4, W/4]

        h = self.us2(h)
        h = self.us2_conv(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h)

        h = self.us1(h)
        h = self.us1_conv(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h)

        return self.out(h)


# =========================
# OU/VP-SDE + loss + sampler
# =========================

@dataclass(frozen=True)
class VPSDE:
    """
    VP SDE (time-inhomogeneous OU):
        dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW

    with linear beta(t) on [0,1].

    Marginal:
        x_t = alpha(t) x_0 + sigma(t) eps
    """
    beta_min: float = 0.1
    beta_max: float = 20.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.int_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - a * a, min=1e-8))


def diffusion_loss_eps(
    model: CondUNetTiny,
    sde: VPSDE,
    x0: torch.Tensor,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    p_uncond: float = 0.1,
    t_power: float = 1.0,
) -> torch.Tensor:
    """
    Eps-prediction denoising loss with conditioning dropout for CFG.

    x0 is expected in [0,1]. Internally we map to [-1,1].

    t_power:
        Sample t as t = u**t_power to bias towards small t (t_power>1).
    """
    device = x0.device
    B = x0.shape[0]

    x0 = x0 * 2.0 - 1.0  # -> [-1,1]

    u = torch.rand((B,), device=device)
    t = u ** float(t_power)

    eps = torch.randn_like(x0)

    a = sde.alpha(t).view(B, 1, 1, 1)
    s = sde.sigma(t).view(B, 1, 1, 1)
    x_t = a * x0 + s * eps

    # CFG training: learn unconditional branch by dropping conditioning sometimes.
    if p_uncond > 0.0:
        drop = (torch.rand((B,), device=device) < p_uncond)
        if drop.any():
            y_cat = y_cat.clone()
            y_cont = y_cont.clone()
            y_cat[drop] = model.n_types  # null token
            y_cont[drop] = 0.0

    eps_hat = model(x_t, t, y_cat, y_cont)
    return F.mse_loss(eps_hat, eps)


@torch.no_grad()
def predict_eps_cfg(
    model: CondUNetTiny,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Classifier-free guidance:
        eps = eps_u + s * (eps_c - eps_u)
    """
    if guidance_scale <= 0.0:
        return model(x_t, t, y_cat, y_cont)

    y_cat_u = torch.full_like(y_cat, fill_value=model.n_types)
    y_cont_u = torch.zeros_like(y_cont)

    eps_u = model(x_t, t, y_cat_u, y_cont_u)
    eps_c = model(x_t, t, y_cat, y_cont)
    return eps_u + guidance_scale * (eps_c - eps_u)


def _probflow_drift(
    model: CondUNetTiny,
    sde: VPSDE,
    x: torch.Tensor,
    t: torch.Tensor,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Probability-flow ODE drift for VP-SDE when training eps-prediction.

    ODE:
        dx = [ -0.5 beta(t) x - 0.5 beta(t) score(x,t) ] dt
    with score = -eps_hat / sigma(t).
    """
    B = x.shape[0]
    beta_t = sde.beta(t).view(B, 1, 1, 1)
    sigma_t = sde.sigma(t).view(B, 1, 1, 1)

    eps_hat = predict_eps_cfg(model, x, t, y_cat, y_cont, guidance_scale=guidance_scale)
    score = -eps_hat / sigma_t

    return -0.5 * beta_t * x - 0.5 * beta_t * score


@torch.no_grad()
def sample_probability_flow_ode(
    model: CondUNetTiny,
    sde: VPSDE,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    img_shape: Tuple[int, int, int, int],
    n_steps: int = 200,
    guidance_scale: float = 0.0,
    t_end: float = 1e-3,
) -> torch.Tensor:
    """
    Deterministic sampler: probability-flow ODE with Heun (2nd order).

    Improvements:
      - Quadratic time grid (more steps near t_end)
      - Final x0 projection from x_{t_end}
    """
    device = y_cat.device
    B, C, H, W = img_shape
    assert C == 1

    t_end = float(t_end)
    if not (0.0 < t_end < 1.0):
        raise ValueError(f"t_end must be in (0,1), got {t_end}")

    # Model is trained on [-1,1], so start from N(0,I) in that space.
    x = torch.randn((B, C, H, W), device=device)

    # Quadratic grid: concentrates steps near t_end (stiff region).
    u = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    ts = t_end + (1.0 - t_end) * (1.0 - u) ** 2

    for i in range(n_steps):
        t = ts[i].expand(B)
        t_next = ts[i + 1].expand(B)
        dt = (t_next - t).view(B, 1, 1, 1)  # negative

        drift = _probflow_drift(model, sde, x, t, y_cat, y_cont, guidance_scale)
        x_euler = x + drift * dt
        drift_next = _probflow_drift(model, sde, x_euler, t_next, y_cat, y_cont, guidance_scale)
        x = x + 0.5 * (drift + drift_next) * dt

    # Final x0 projection from x_{t_end}.
    t_final = ts[-1].expand(B)
    a = sde.alpha(t_final).view(B, 1, 1, 1)
    s = sde.sigma(t_final).view(B, 1, 1, 1)
    eps_hat = predict_eps_cfg(model, x, t_final, y_cat, y_cont, guidance_scale=guidance_scale)
    x0_hat = (x - s * eps_hat) / torch.clamp(a, min=1e-6)

    # Map back to [0,1] for visualisation.
    x0 = (x0_hat + 1.0) * 0.5
    return x0.clamp(0.0, 1.0)

