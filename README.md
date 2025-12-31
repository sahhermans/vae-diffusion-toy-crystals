# Toy Crystals — conditional VAE and latent-prior experiments

Small, reproducible sandbox for conditional generative modelling on a synthetic “toy-crystals” dataset (periodic lattices rendered as Gaussian “atoms”).

It includes:
- a **conditional VAE** conditioned on **lattice type** (categorical) and **rotation** (continuous),
- baseline latent sampling via **\(z \sim N(0,I)\)** and a **mixture-of-posteriors (MoP)** / aggregated-posterior proxy,
- a **latent diffusion prior** (DDPM-style noise-prediction objective, DDIM sampling) trained in latent space.

## Key results (saved images)

Representative figures are committed under `assets/`:

- `assets/preview_toycrystals.png`  
  Dataset preview.

- `assets/cond_withrot/`  
  Main experiment: conditional VAE on `(type, theta)` with:
  - `vae_recon.png` — reconstructions (sanity check: encoder/decoder work)
  - `vae_samples_prior.png` — samples from standard VAE prior \(N(0,I)\)
  - `vae_samples_mop.png` — samples from MoP / aggregated posterior proxy
  - `vae_loss.png` — training curve

- `assets/diffusion_firstattempt/`  
  Latent diffusion prior trained on cached encoder latents:
  - `diffusion_samples.png` — diffusion-prior samples (decoded through the VAE)
  - `diffusion_loss.png` — training curve

### Qualitative takeaways
- **Reconstructions** are crisp and confirm the conditional VAE learns the data manifold.
- **\(N(0,I)\) prior sampling** often leaves the manifold (blurry / incorrect structure), indicating a prior mismatch.
- **MoP sampling** produces much cleaner outputs (sampling from encoder posteriors of real datapoints).
- **Diffusion prior** samples are typically **more consistent** than the standard prior samples, but do not yet reach the MoP baseline in this configuration.

## Dataset

Each datapoint is an image `x` of shape `[1, H, W]` plus conditioning:
- `y_cat`: lattice type (int)
- `y_cont`: continuous vector including `theta` (rotation angle).  
  In `rot_only` mode: `y_cont = [0, theta, 0, 0]`.

Two ways to obtain data:
1) On-the-fly generation for quick iteration.
2) Precomputed `.pt` files on disk for faster training and reproducibility.

## Scripts

- `scripts/preview_data.py` — quick dataset preview (writes to `results/`)
- `scripts/build_dataset.py` — generate and save a disk dataset (`.pt`)
- `scripts/train_vae.py` — train VAE and export reconstructions / prior samples / MoP samples
- `scripts/train_diffusion_prior.py` — build latent cache, train diffusion prior, sample and decode

## Installation

This project uses PyTorch, which is **not** pinned in `pyproject.toml` (so install it separately first).

```bash
python -m venv .venv
# activate venv
pip install -U pip

# install PyTorch (CPU/CUDA) using the official instructions for your system
pip install torch

# install this package
pip install -e .
```

## Quickstart
```
# 1) (optional) preview on-the-fly data
python scripts/preview_data.py

# 2) build a reproducible training dataset on disk
python scripts/build_dataset.py --out data/toycrystals_train_rotonly.pt --n-samples 50000 --img-size 64 --n-types 4

# 3) train the (conditional) VAE (use --uncond for an unconditional baseline)
python scripts/train_vae.py --data-path data/toycrystals_train_rotonly.pt --epochs 25

# 4) train the latent diffusion prior and sample (decoded through the VAE)
python scripts/train_diffusion_prior.py --epochs 200
```