from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from toycrystals.disk_data import ToyCrystalsDiskDataset
from toycrystals.models.vae import CondVAE
from toycrystals.models.diffusion_prior import DiffusionPrior, DiffusionSchedule, DiffusionPriorFiLM


@torch.no_grad()
def build_latent_dataset(
    vae: CondVAE,
    dl: DataLoader,
    device: torch.device,
    z_target: str = "mu",  # "mu" or "sample"
    max_items: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns tensors (z0, y_cat, y_cont).
    """
    vae.eval()

    zs = []
    ycats = []
    yconts = []

    seen = 0
    for x, y_cat, y_cont in tqdm(dl, desc="encoding latents"):
        x = x.to(device)
        y_cat = y_cat.to(device)
        y_cont = y_cont.to(device)

        mu, logvar = vae.encode(x, y_cat, y_cont)
        if z_target == "mu":
            z0 = mu
        elif z_target == "sample":
            z0 = vae.reparameterise(mu, logvar)
        else:
            raise ValueError(f"unknown z_target={z_target}")

        zs.append(z0.detach().cpu())
        ycats.append(y_cat.detach().cpu())
        yconts.append(y_cont.detach().cpu())

        seen += x.shape[0]
        if max_items is not None and seen >= max_items:
            break

    z0 = torch.cat(zs, dim=0)
    y_cat = torch.cat(ycats, dim=0)
    y_cont = torch.cat(yconts, dim=0)
    return z0, y_cat, y_cont

@torch.no_grad()
def save_diffusion_samples(
    vae: CondVAE,
    prior: DiffusionPrior,
    sched: DiffusionSchedule,
    out_path: str,
    device: torch.device,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    n: int = 36,
    theta_max: float = math.pi / 3.0,
    ddim_steps: int = 50,
) -> None:
    """
    Make a 6x6 grid, cycling lattice types, sweeping theta in [0, pi/3].
    Diffusion samples in standardised latent space; we unstandardise before decoding.
    """
    vae.eval()
    prior.eval()

    y_cat = torch.tensor([i % vae.n_types for i in range(n)], device=device, dtype=torch.int64)

    thetas = torch.linspace(0.0, theta_max, steps=n, device=device)
    y_cont = torch.zeros((n, vae.y_cont_dim), device=device)
    y_cont[:, 1] = thetas  # rot_only uses theta at index 1

    # Sample in normalised latent space
    z_norm = sched.ddim_sample(prior, y_cat=y_cat, y_cont=y_cont, n_steps=ddim_steps, eta=0.0)

    # Map back to VAE latent space
    z_mean = z_mean.to(device)
    z_std = z_std.to(device)
    z = z_norm * z_std + z_mean

    x = vae.decode(z, y_cat, y_cont)

    fig, axes = plt.subplots(6, 6, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i, 0].cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t={int(y_cat[i].item())}", fontsize=7)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-path", type=str, default="data/toycrystals_train_rotonly.pt")
    # Load frozen VAE
    p.add_argument("--vae-ckpt", type=str, default="checkpoints/vae_last.pt")
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--y-cont-dim", type=int, default=4)
    # Latent dataset
    p.add_argument("--z-target", type=str, choices=["mu", "sample"], default="mu")
    p.add_argument("--latent-cache", type=str, default="data/latents_rotonly_mu.pt")
    p.add_argument("--rebuild-latents", action="store_true")
    p.add_argument("--max-items", type=int, default=50_000)
    # Diffusion
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=1)
    p.add_argument("--t-emb-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-4)
    # Sampling
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--prior-ckpt", type=str, default="checkpoints/diffusion_prior_last.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample-only", action="store_true") 
    args = p.parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available; using cpu")
    else:
        device = torch.device(args.device)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # --- load dataset ---
    ds = ToyCrystalsDiskDataset(args.data_path)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, drop_last=False)

    # --- load frozen VAE ---
    vae = CondVAE(z_dim=args.z_dim, n_types=args.n_types, y_cont_dim=args.y_cont_dim, cond_drop=0.0).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    # --- build / load latent dataset ---
    if (not args.rebuild_latents) and os.path.exists(args.latent_cache):
        obj = torch.load(args.latent_cache, map_location="cpu")
        z0 = obj["z0"]
        y_cat = obj["y_cat"]
        y_cont = obj["y_cont"]

        if "z_mean" in obj and "z_std" in obj:
            z_mean = obj["z_mean"]
            z_std = obj["z_std"]
        else:
            z_mean = z0.mean(dim=0, keepdim=True)
            z_std = torch.clamp(z0.std(dim=0, keepdim=True), min=1e-6)

        print(f"loaded latents: {args.latent_cache}  z0={tuple(z0.shape)}")
    else:
        z0, y_cat, y_cont = build_latent_dataset(
            vae, dl, device=device, z_target=args.z_target, max_items=args.max_items
        )
        z_mean = z0.mean(dim=0, keepdim=True)
        z_std = torch.clamp(z0.std(dim=0, keepdim=True), min=1e-6)

        torch.save(
            {"z0": z0, "y_cat": y_cat, "y_cont": y_cont, "z_mean": z_mean, "z_std": z_std},
            args.latent_cache,
        )
        print(f"saved latents: {args.latent_cache}  z0={tuple(z0.shape)}")

    # Standardise for diffusion training
    z0_norm = (z0 - z_mean) / z_std

    latent_ds = TensorDataset(z0_norm, y_cat, y_cont)
    latent_dl = DataLoader(latent_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # --- diffusion prior ---
    prior = DiffusionPriorFiLM(
        z_dim=args.z_dim,
        n_types=args.n_types,
        y_cont_dim=args.y_cont_dim,
        t_emb_dim=args.t_emb_dim,
        width=args.width,
        n_blocks=8,         
        y_cat_emb_dim=64,   
    ).to(device)

    # --- diffusion schedule ---
    sched = DiffusionSchedule.linear(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    # Load prior weights only if we explicitly want to resume, or if we're sampling.
    if (args.sample_only or args.resume) and os.path.exists(args.prior_ckpt):
        prior.load_state_dict(torch.load(args.prior_ckpt, map_location=device))
        print(f"loaded diffusion prior: {args.prior_ckpt}")
    
    if args.sample_only:
        save_diffusion_samples(
            vae=vae,
            prior=prior,
            sched=sched,
            out_path="results/diffusion_samples.png",
            device=device,
            z_mean=z_mean,
            z_std=z_std,
            ddim_steps=args.ddim_steps,
        )
        print("sample-only: saved results/diffusion_samples.png")
        return 0

    opt = torch.optim.Adam(prior.parameters(), lr=args.lr)

    loss_hist = []
    print("starting diffusion training loop.")
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    for epoch in range(args.epochs):
        
        bucket_sum = torch.zeros(4, device=device)
        bucket_n = torch.zeros(4, device=device)

        prior.train()
        total = 0.0

        for z0n_b, y_cat_b, y_cont_b in latent_dl:
            z0n_b = z0n_b.to(device)
            y_cat_b = y_cat_b.to(device)
            y_cont_b = y_cont_b.to(device)

            B = z0n_b.shape[0]
            # Bias towards small t (harder, affects final sample quality)
            u = torch.rand((B,), device=device)
            t = torch.clamp((u ** 2 * args.T).long(), 0, args.T - 1)
            
            eps = torch.randn_like(z0n_b)
            z_t = sched.q_sample(z0=z0n_b, t=t, eps=eps)

            eps_pred = prior(z_t, t, y_cat_b, y_cont_b)
            loss = torch.mean((eps_pred - eps) ** 2)

            per = ((eps_pred - eps) ** 2).mean(dim=1)  # [B]
            # 4 buckets over t
            q = torch.clamp((t.float() / args.T * 4).long(), 0, 3)  # [B]
            for b in range(4):
                m = (q == b)
                if torch.any(m):
                    bucket_sum[b] += per[m].sum()
                    bucket_n[b] += m.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())

        avg = total / len(latent_dl)
        loss_hist.append(avg)
        print(f"epoch {epoch+1:02d}/{args.epochs} diffusion_loss={avg:.6f}")

        torch.save(prior.state_dict(), "checkpoints/diffusion_prior_last.pt")

        # quick visual every epoch
        save_diffusion_samples(
            vae=vae,
            prior=prior,
            sched=sched,
            out_path="results/diffusion_samples.png",
            device=device,
            z_mean=z_mean,
            z_std=z_std,
            ddim_steps=args.ddim_steps,
        )

        bucket_avg = (bucket_sum / torch.clamp(bucket_n, min=1)).detach().cpu().tolist()
        print("  bucket loss (low t -> high t):", [f"{v:.3f}" for v in bucket_avg])


    # loss curve
    fig = plt.figure(figsize=(5, 3))
    plt.plot(loss_hist, label="diffusion_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/diffusion_loss.png", dpi=200)
    plt.close(fig)

    print("saved: results/diffusion_samples.png, results/diffusion_loss.png, checkpoints/diffusion_prior_last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
