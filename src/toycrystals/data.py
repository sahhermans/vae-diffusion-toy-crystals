from __future__ import annotations

import math
from typing import Tuple

import torch
from torch.utils.data import Dataset


def _uniform(g: torch.Generator, low: float, high: float) -> float:
    """Deterministic uniform draw using the provided generator."""
    return float((low + (high - low) * torch.rand((), generator=g)).item())


def _rotation_matrix(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _lattice_definition(
    lattice_type: int,
    a: float,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns lattice vectors v1, v2 and a basis (K,2) in pixel coordinates.

    lattice_type:
      0 = square
      1 = rectangular (random aspect)
      2 = triangular
      3 = honeycomb (triangular + 2-atom basis)
    """
    a = float(a)

    if lattice_type == 0:  # square
        v1 = torch.tensor([a, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.0, a], dtype=torch.float32)
        basis = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    elif lattice_type == 1:  # rectangular
        aspect = _uniform(g, 0.75, 1.35) # nuisance variable not measured/conditioned on
        b = a * aspect
        v1 = torch.tensor([a, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.0, b], dtype=torch.float32)
        basis = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    elif lattice_type == 2:  # triangular
        v1 = torch.tensor([a, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.5 * a, (math.sqrt(3) / 2.0) * a], dtype=torch.float32)
        basis = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    elif lattice_type == 3:  # honeycomb: triangular lattice with 2-point basis
                             # Current approach leads to different atom densities 
                             # than other lattice types. Address later if needed.
        v1 = torch.tensor([a, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.5 * a, (math.sqrt(3) / 2.0) * a], dtype=torch.float32)
        basis = torch.tensor(
            [
                [0.0, 0.0],
                [0.5 * a, (math.sqrt(3) / 6.0) * a],
            ],
            dtype=torch.float32,
        )

    else:
        raise ValueError(f"Unknown lattice_type={lattice_type}")

    return v1, v2, basis


def _make_points(
    lattice_type: int,
    a: float,
    H: int,
    W: int,
    theta: float,
    vacancy: float,
    jitter: float,
    g: torch.Generator,
) -> torch.Tensor:
    """
    Generate atom centres (N,2) in pixel coordinates (x,y).
    """
    v1, v2, basis = _lattice_definition(lattice_type=lattice_type, a=a, g=g)

    # Centre of the image in pixel coordinates.
    centre = torch.tensor([W / 2.0, H / 2.0], dtype=torch.float32)

    # Build a grid of lattice indices large enough to cover the canvas.
    # We over-generate then crop; simpler and robust after rotation/jitter.
    margin = 2.0 * a
    extent = max(H, W) + margin
    n1 = int(math.ceil(extent / float(v1.norm().item()))) + 2
    n2 = int(math.ceil(extent / float(v2.norm().item()))) + 2
    # Note that n1, n2 ignore basis size, but margin are assumed to cover that.

    # Generate points: i*v1 + j*v2 + basis_k, then centre.
    pts = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            base = i * v1 + j * v2
            for k in range(basis.shape[0]):
                pts.append(base + basis[k])

    P = torch.stack(pts, dim=0)  # [N,2] around origin
    P = P + centre  # shift to image centre

    # Apply rotation around the centre.
    R = _rotation_matrix(theta)
    P = (P - centre) @ R.T + centre

    # Drop random points (vacancies).
    if vacancy > 0.0:
        keep = torch.rand((P.shape[0],), generator=g) > vacancy
        P = P[keep]

    # Add Gaussian jitter.
    if jitter > 0.0:
        P = P + torch.randn(P.shape, generator=g) * jitter

    # Crop to slightly beyond the image bounds (to keep edge atoms).
    x = P[:, 0]
    y = P[:, 1]
    keep = (x > -margin) & (x < W + margin) & (y > -margin) & (y < H + margin)
    return P[keep]


def _render_gaussians(points_xy: torch.Tensor, H: int, W: int, sigma: float) -> torch.Tensor:
    """
    Render a sum of isotropic Gaussians centred at points.
    points_xy: [N,2] in (x,y) pixel coordinates.
    Returns: [H,W] float32 (not yet normalised).
    """
    if points_xy.numel() == 0:
        return torch.zeros((H, W), dtype=torch.float32)

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )  # yy/xx: [H,W]

    P = points_xy.to(dtype=torch.float32)
    dx = xx[None, :, :] - P[:, 0][:, None, None]
    dy = yy[None, :, :] - P[:, 1][:, None, None]

    img = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).sum(dim=0)
    return img


class ToyCrystalsDataset(Dataset):
    """
    Synthetic dataset of "toy crystals".

    Each item is deterministically generated from (seed + idx), so no files are needed.
    """
    def __init__(self, n_samples: int = 50_000, img_size: int = 64, seed: int = 0, n_types: int = 4, simple: bool = False, rot_only: bool = False) -> None:
        self.n_samples = int(n_samples)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.n_types: int = n_types
        self.simple: bool = simple
        self.rot_only: bool = rot_only

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))

        H = W = self.img_size

        lattice_type = int(torch.randint(0, self.n_types, (1,), generator=g).item())
        a = _uniform(g, 6.0, 14.0)                       # lattice spacing (pixels)
        theta = _uniform(g, 0.0, math.pi / 3.0)          # rotation (radians)
        vacancy = _uniform(g, 0.0, 0.25)                 # fraction removed
        jitter = _uniform(g, 0.0, 0.6)                   # jitter std in pixels

        if self.simple:
            a = 10.0
            theta = 0.0
            vacancy = 0.0
            jitter = 0.0

        if self.rot_only:
            a = 10.0
            vacancy = 0.0
            jitter = 0.0

        points = _make_points(
            lattice_type=lattice_type,
            a=a,
            H=H,
            W=W,
            theta=theta,
            vacancy=vacancy,
            jitter=jitter,
            g=g,
        )

        # Tie atom blur to spacing for nicer visuals across 'a'.
        sigma_atom = max(0.6, 0.12 * a)
        img = _render_gaussians(points_xy=points, H=H, W=W, sigma=sigma_atom)

        # Normalise to [0,1].
        img = img / (img.max() + 1e-8)
        img = img.clamp(0.0, 1.0)

        x = img[None, :, :].to(dtype=torch.float32)  # [1,H,W]
        y_cat = torch.tensor(lattice_type, dtype=torch.int64)
        if self.simple:
            y_cont = torch.zeros(4, dtype=torch.float32)
        elif self.rot_only:
            y_cont = torch.tensor([0.0, theta, 0.0, 0.0], dtype=torch.float32)
        else:
            y_cont = torch.tensor([a, theta, vacancy, jitter], dtype=torch.float32)

        return x, y_cat, y_cont
