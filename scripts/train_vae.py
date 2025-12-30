from __future__ import annotations

from typing import Optional
import math

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from toycrystals.data import ToyCrystalsDataset
from toycrystals.models.vae import CondVAE, VAE


def kl_stats(mu: torch.Tensor, 
             logvar: torch.Tensor, 
             free_bits: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (kl_used_for_loss, kl_raw), both averaged over batch.
    free_bits is in nats per latent dimension.
    """
    # [B, z_dim]
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)

    kl_raw = kl_dim.sum(dim=1).mean()

    if free_bits > 0.0:
        fb = torch.tensor(free_bits, device=kl_dim.device, dtype=kl_dim.dtype)
        kl_used = torch.maximum(kl_dim, fb).sum(dim=1).mean()
    else:
        kl_used = kl_raw

    return kl_used, kl_raw


@torch.no_grad()
def save_recon_grid(
    model: CondVAE,
    x: torch.Tensor,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    out_path: str,
    n_pairs: int = 16,
    uncond: bool = False
) -> None:
    model.eval()

    if uncond:
        x_hat, _, _ = model(x)
    else:
        x_hat, _, _ = model(x, y_cat, y_cont)

    n = min(n_pairs, x.shape[0])
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))
    axes = list(axes.flat)

    for i in range(n):
        t = int(y_cat[i].item())  # label only

        axes[2 * i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[2 * i].set_title(f"X (type={t})")
        axes[2 * i].axis("off")

        axes[2 * i + 1].imshow(x_hat[i, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[2 * i + 1].set_title(f"X̂ (type={t})")
        axes[2 * i + 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_prior_samples(model: CondVAE, 
                       out_path: str, 
                       device: torch.device, 
                       uncond: bool, 
                       theta_max: float = math.pi / 3.0
) -> None:
    model.eval()

    n = 36
    z = torch.randn((n, model.z_dim), device=device)

    if uncond:
        x = model.decode(z)
    else:
        # Cycle lattice types for a mixed grid
        y_cat = torch.tensor([i % model.n_types for i in range(n)], device=device, dtype=torch.int64)

        thetas = torch.linspace(0.0, theta_max, steps=n, device=device)  # adjust max as needed
        y_cont = torch.zeros((n, 4), device=device)
        if model.y_cont_dim <= 1:
            raise ValueError("Expected y_cont to have theta at index 1.")
        y_cont[:, 1] = thetas

        x = model.decode(z, y_cat, y_cont)  

    fig, axes = plt.subplots(6, 6, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        if not uncond:
            ax.set_title(f"t={int(y_cat[i].item())}", fontsize=7)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



@torch.no_grad()
def save_mop_samples(
    model: CondVAE,
    dl: DataLoader,
    out_path: str,
    device: torch.device,
    uncond: bool,
    pool_size: int = 4096,
    theta_max: float = math.pi / 3.0,
    decode_with_target: bool = True,
) -> None:
    """
    Mixture-of-posteriors sampling baseline.

    For each target condition (lattice type, theta) on a fixed grid:
      - pick a real example from a candidate pool with same type and nearest theta
      - encode it -> (mu, logvar)
      - sample z ~ N(mu, sigma^2) via reparameterise
      - decode using either the target condition (decode_with_target=True) or the matched example's condition
    """
    model.eval()

    n = 36

    # Build the same "fixed condition" grid as save_prior_samples
    if uncond:
        # (no conditioning.)
        pass
    else:
        y_target_cat = torch.tensor([i % model.n_types for i in range(n)], device=device, dtype=torch.int64)
        thetas = torch.linspace(0.0, theta_max, steps=n, device=device)
        y_target_cont = torch.zeros((n, model.y_cont_dim), device=device)
        y_target_cont[:, 1] = thetas

    # --- collect a candidate pool from the dataloader (on device) ---
    xs, ycats, yconts = [], [], []
    seen = 0
    for x, y_cat, y_cont in dl:
        xs.append(x)
        ycats.append(y_cat)
        yconts.append(y_cont)
        seen += x.shape[0]
        if seen >= pool_size:
            break

    x_pool = torch.cat(xs, dim=0)[:pool_size].to(device)
    ycat_pool = torch.cat(ycats, dim=0)[:pool_size].to(device)
    ycont_pool = torch.cat(yconts, dim=0)[:pool_size].to(device)

    if uncond:
        # pick n random items from the pool
        idx = torch.randint(0, x_pool.shape[0], (n,), device=device)
        x_sel = x_pool[idx]

        mu, logvar = model.encode(x_sel)
        z = model.reparameterise(mu, logvar)
        x_gen = model.decode(z)

    else:
        # For each target cell, find nearest in pool with same type and closest theta
        idxs = []
        for i in range(n):
            t = y_target_cat[i]
            theta_t = y_target_cont[i, 1]

            mask = (ycat_pool == t)
            if not torch.any(mask):
                # fallback: pick any random index (should not happen with decent pool_size)
                idxs.append(int(torch.randint(0, x_pool.shape[0], (1,), device=device).item()))
                continue

            pool_idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            dtheta = (ycont_pool[pool_idxs, 1] - theta_t).abs()
            best_local = int(torch.argmin(dtheta).item())
            idxs.append(int(pool_idxs[best_local].item()))

        idx = torch.tensor(idxs, device=device, dtype=torch.long)

        x_sel = x_pool[idx]
        y_sel_cat = ycat_pool[idx]
        y_sel_cont = ycont_pool[idx]

        # Sample z from q(z|x_sel, y_sel)
        mu, logvar = model.encode(x_sel, y_sel_cat, y_sel_cont)
        z = model.reparameterise(mu, logvar)

        # Decode either with the fixed target condition (for a true fixed grid),
        # or with the matched example condition (slightly more self-consistent).
        if decode_with_target:
            x_gen = model.decode(z, y_target_cat, y_target_cont)
            y_show = y_target_cat
        else:
            x_gen = model.decode(z, y_sel_cat, y_sel_cont)
            y_show = y_sel_cat

    # --- plot ---
    fig, axes = plt.subplots(6, 6, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_gen[i, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        if not uncond:
            ax.set_title(f"t={int(y_show[i].item())}", fontsize=7)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--y-cont-dim", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.0003)
    p.add_argument("--device", type=str, default="cuda")  # "cuda" or "cpu"
    p.add_argument("--num-workers", type=int, default=0)  # keep 0 on Windows unless you want to tune it
    p.add_argument("--data-path", type=str, default="data/toycrystals_train_rotonly.pt")
    p.add_argument("--cond-drop", type=float, default=0.0)
    p.add_argument("--uncond", dest="uncond", action="store_true", help="Train unconditional VAE.")
    p.add_argument("--cond", dest="uncond", action="store_false", help="Train conditional VAE.")
    p.add_argument("--free-bits", type=float, default=0.05, help="Free bits threshold in nats per latent dim (0 disables).")
    p.set_defaults(uncond=False)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available; using cpu")
    else:
        device = torch.device(args.device)

    if args.data_path:
        from toycrystals.disk_data import ToyCrystalsDiskDataset
        ds = ToyCrystalsDiskDataset(args.data_path)
    else:
        ds = ToyCrystalsDataset(n_samples=args.n_samples, img_size=args.img_size, seed=args.seed)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda")
    )

    if args.uncond:
        model = VAE(z_dim=args.z_dim).to(device)
    else:
        print("Training conditional VAE")
        model = CondVAE(z_dim=args.z_dim, 
                        n_types=args.n_types, 
                        y_cont_dim=args.y_cont_dim, 
                        cond_drop=args.cond_drop
                        ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_hist = []
    recon_hist = []
    kl_hist = []
    klr_hist = []

    print("starting training loop...")
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_klr = 0.0

        for x, y_cat, y_cont in dl:
            x = x.to(device)
            y_cat = y_cat.to(device)
            y_cont = y_cont.to(device)

            if args.uncond:
                x_hat, mu, logvar = model(x)
            else:
                x_hat, mu, logvar = model(x, y_cat, y_cont)

            recon = torch.mean((x_hat - x) ** 2)
            kl_used, kl_raw = kl_stats(mu, logvar, free_bits=args.free_bits)
            beta = args.beta * min(1.0, (epoch + 1) / 5.0)
            loss = recon + beta * kl_used

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl_used.item())
            total_klr += float(kl_raw.item())

        n_batches = len(dl)
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        avg_klr = total_klr / n_batches

        loss_hist.append(avg_loss)
        recon_hist.append(avg_recon)
        kl_hist.append(avg_kl)
        klr_hist.append(avg_klr)

        print(f"epoch {epoch+1:02d}/{args.epochs} loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.6f}")

        torch.save(model.state_dict(), "checkpoints/vae_last.pt")

    # Diagnostics (use one fresh batch)
    x0, y0_cat, y0_cont = next(iter(dl))
    x0 = x0[:16].to(device)
    y0_cat = y0_cat[:16].to(device)
    y0_cont = y0_cont[:16].to(device)

    save_recon_grid(model, x0, y0_cat, y0_cont, "results/vae_recon.png", uncond=args.uncond)
    save_prior_samples(model, "results/vae_samples_prior.png", device=device, uncond=args.uncond)

    save_mop_samples(model, dl, "results/vae_samples_mop.png", device=device, uncond=args.uncond, pool_size=4096, decode_with_target=True)

    # Loss curves
    fig = plt.figure(figsize=(5, 3))
    plt.plot(loss_hist, label="total")
    plt.plot(recon_hist, label="recon")
    plt.plot(kl_hist, label="kl")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/vae_loss.png", dpi=200)
    plt.close(fig)

    print("saved: results/vae_recon.png, results/vae_samples_prior.png, results/vae_loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
