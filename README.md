# Toy Crystals: Conditional VAE + Latent Diffusion Prior

This repo implements:
1) a **conditional VAE** trained on a synthetic "toy crystals" dataset (periodic lattices with Gaussian atoms),
2) a **mixture-of-posteriors baseline** for sampling latents,
3) a **latent diffusion prior** (DDPM training, DDIM sampling) to generate latents that decode well.

## Why
A vanilla VAE samples latents from a simple prior (usually N(0, I)). In practice, the latents produced by the encoder
(the aggregated posterior) can differ from that prior, which can hurt sample quality. This repo compares:
- VAE prior sampling (z ~ N(0, I))
- mixture baseline sampling (z ~ 1/N Σ N(mu_n, sigma_n^2))
- diffusion prior sampling (z ~ learned p(z|y))

## Quickstart
### 1) Install PyTorch (CUDA)
Install a CUDA-enabled PyTorch build appropriate for your system.

### 2) Install this repo
```bash
python -m pip install -e ".[dev]"
```



# Simple uncond results made with
```bash
python scripts/train_vae.py --uncond --beta 0.0005 --z-dim 16 --epochs 15 --data-path data/toycrystals_train_simple.pt
```

# Simple cond results made with
```bash
python scripts/train_vae.py --cond --beta 0.0005 --z-dim 32 --epochs 15  --free-bits 0.02 --data-path data/toycrystals_train_simple.pt
```

# Cond rotonly results made with
```bash
python scripts/train_vae.py --cond --beta 0.0003 --z-dim 32 --epochs 15  --free-bits 0.05 --data-path data/toycrystals_train_rotonly.pt
```