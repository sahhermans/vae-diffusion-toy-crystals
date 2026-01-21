from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime

from toycrystals.disk_data import ToyCrystalsDiskDataset
from toycrystals.models.sde_score_model import (
    CondUNetTiny,
    VPSDE,
    diffusion_loss_eps,
    sample_probability_flow_ode,
    save_sde_samples
)


def _make_run_name(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # keep it short but informative (edit if you want)
    return (
        f"{ts}_lr{args.lr:.2e}_ch{args.base_ch}"
        f"_b{args.beta_max:g}_tp{args.t_power:g}_pu{args.p_uncond:g}"
    )


def _save_checkpoint(
    ckpt_path: str,
    *,
    epoch_next: int,
    model: CondUNetTiny,
    opt: torch.optim.Optimizer,
    loss_hist: list[float],
    config: dict[str, Any],
    ema_model: CondUNetTiny | None = None,
) -> None:
    check_di: dict[str, Any] = {
            "epoch_next": int(epoch_next),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "loss_hist": list(loss_hist),
            "config": dict(config),
        }
    if ema_model is not None:
        check_di["ema"] = ema_model.state_dict()
    torch.save(check_di, ckpt_path)


def _try_load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    model: CondUNetTiny,
    opt: torch.optim.Optimizer,
    ema_model: CondUNetTiny | None = None, 
) -> tuple[int, list[float]]:
    if not os.path.exists(ckpt_path):
        return 0, []
    obj: dict[str, Any] = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model"])
    opt.load_state_dict(obj["opt"])
    
    if ema_model is not None:
        if "ema" in obj:
            ema_model.load_state_dict(obj["ema"])
        else:
            # backwards compat: if older ckpt has no EMA, initialise EMA from current model
            ema_model.load_state_dict(model.state_dict())
  
    epoch_next = int(obj.get("epoch_next", 0))
    loss_hist = list(obj.get("loss_hist", []))
    return epoch_next, loss_hist


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-path", type=str, default="data/toycrystals_train_rotonly.pt")

    # Output
    p.add_argument("--out-dir", type=str, default=None, help="Run output directory. If omitted, a timestamped run dir is created under runs/sde_score/")
    p.add_argument("--resume", action="store_true")

    # Model
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--y-cont-dim", type=int, default=4)
    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--cond-ch", type=int, default=8)
    p.add_argument("--time-ch", type=int, default=8)

    # SDE schedule
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=30.0)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument(
        "--t-power",
        type=float,
        default=1.0,
        help="Sample t as t=u**t_power. >1 biases towards small t.",
    )
    p.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA. Typical: 0.999 or 0.9999")

    # Sampling during training
    p.add_argument("--sample-every", type=int, default=10000)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--cfg", type=float, default=0)
    p.add_argument("--t-end", type=float, default=1e-3)
    p.add_argument("--sample-from-ema", type=int, default=1, choices=[0, 1], help="If EMA enabled, save sample grids using EMA weights.")

    args = p.parse_args()
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        base = os.path.join("runs", "sde_score")
        args.out_dir = os.path.join(base, _make_run_name(args))
    print(f"run dir: {args.out_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available; using cpu")
    else:
        device = torch.device(args.device)

    results_dir = os.path.join(args.out_dir, "results")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    ckpt_path = os.path.join(ckpt_dir, "sde_score_model_last.pt")

    # --- dataset ---
    ds = ToyCrystalsDiskDataset(args.data_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # --- model + sde ---
    model = CondUNetTiny(
        n_types=args.n_types,
        y_cont_dim=args.y_cont_dim,
        base_ch=args.base_ch,
        emb_dim=args.emb_dim,
        cond_ch=args.cond_ch,
        time_ch=args.time_ch,
    ).to(device)

    ema_model = None
    if args.ema_decay > 0.0:
        if not (0.0 < args.ema_decay < 1.0):
            raise ValueError("--ema-decay must be in (0,1) or 0 to disable.")
        ema_model = CondUNetTiny(
            n_types=args.n_types,
            y_cont_dim=args.y_cont_dim,
            base_ch=args.base_ch,
            emb_dim=args.emb_dim,
            cond_ch=args.cond_ch,
            time_ch=args.time_ch,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for p_ in ema_model.parameters():
            p_.requires_grad_(False)

    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max)

    config = {
    "img_ch": 1,
    "n_types": args.n_types,
    "y_cont_dim": args.y_cont_dim,
    "base_ch": args.base_ch,
    "emb_dim": args.emb_dim,
    "cond_ch": args.cond_ch,
    "time_ch": args.time_ch,
    "beta_min": args.beta_min,
    "beta_max": args.beta_max,
    # stored for reproducibility / bookkeeping (even if sampling doesn’t use it)
    "t_power": args.t_power,
    "p_uncond": args.p_uncond,
             }
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    loss_hist: list[float] = []
    if args.resume:
        start_epoch, loss_hist = _try_load_checkpoint(ckpt_path, device=device, model=model, opt=opt, ema_model=ema_model)
        if start_epoch > 0:
            print(f"resumed from: {ckpt_path} (next epoch {start_epoch+1})")

    print("starting SDE score-model training loop.")
    if torch.cuda.is_available() and device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    # Ensure metrics file exists (append-only, one JSON per epoch).
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as _:
            pass

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0

        it = tqdm(dl, desc=f"epoch {epoch+1:03d}/{args.epochs}")
        for x0, y_cat, y_cont in it:
            x0 = x0.to(device)
            y_cat = y_cat.to(device)
            y_cont = y_cont.to(device)

            loss = diffusion_loss_eps(
                model=model,
                sde=sde,
                x0=x0,
                y_cat=y_cat,
                y_cont=y_cont,
                p_uncond=args.p_uncond,
                t_power=args.t_power,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if ema_model is not None:
                decay = float(args.ema_decay)
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

            total += float(loss.item())
            it.set_postfix(loss=float(loss.item()))

        avg = total / len(dl)
        loss_hist.append(avg)
        print(f"epoch {epoch+1:03d}/{args.epochs}: loss={avg:.6f}")

        # Save checkpoint + metrics for sweep/resume.
        _save_checkpoint(
                    ckpt_path,
                    epoch_next=epoch + 1,
                    model=model,
                    opt=opt,
                    loss_hist=loss_hist,
                    config=config,
                    ema_model=ema_model, 
                        )
        with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch + 1, "loss": avg}) + "\n")

        # Save samples every N epochs and always at the end.
        if ((epoch + 1) % args.sample_every == 0) or (epoch == args.epochs - 1):
            out_path = os.path.join(results_dir, f"sde_samples_epoch_{epoch+1:03d}.png")
            
            sample_model = model
            if (ema_model is not None) and (args.sample_from_ema == 1):
                sample_model = ema_model                  
            
            save_sde_samples(
                model=sample_model,
                sde=sde,
                out_path=out_path,
                device=device,
                steps=args.sample_steps,
                cfg=args.cfg,
                t_end=args.t_end,
            )
            print(f"  saved: {out_path}")

    # Loss curve (one point per epoch).
    fig = plt.figure(figsize=(5, 3))
    plt.plot(loss_hist, label="eps_mse")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    loss_png = os.path.join(results_dir, "sde_loss.png")
    plt.savefig(loss_png, dpi=200)
    plt.close(fig)

    print(f"saved: {loss_png}")
    print(f"checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
