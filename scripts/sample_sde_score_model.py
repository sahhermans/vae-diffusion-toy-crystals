#!/usr/bin/env python3
import argparse
import os
import math
import torch

from toycrystals.models.sde_score_model import CondUNetTiny, VPSDE, save_sde_samples

# example call
# python scripts/sample_sde_score_model.py \
#   --device cuda \
#   --out-dir runs/sde_score/checkpoints/results \
#   --ckpt runs/sde_score/checkpoints/sde_score_model_t1_96ch.pt \
#   --steps 400 \
#   --cfg 1.5 \
#   --t-end 0.001


def _infer_ckpt_path(out_dir: str, ckpt: str) -> str:
    # allow either a direct path or "last"/"best"
    if ckpt.endswith(".pt"):
        return ckpt
    if ckpt == "last":
        return os.path.join(out_dir, "checkpoints", "sde_score_model_last.pt")
    if ckpt == "best":
        return os.path.join(out_dir, "checkpoints", "sde_score_model_best.pt")
    raise ValueError("ckpt must be a .pt path or one of: last, best")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out-dir", required=True, help="Training output dir containing checkpoints/")
    p.add_argument("--ckpt", default="last", help="Checkpoint: last, best, or path/to/file.pt")

    # sampling knobs
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--cfg", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=1e-3)
    p.add_argument("--theta-max", type=float, default=math.pi / 3.0)
    p.add_argument("--n", type=int, default=36)
    p.add_argument("--use-ema", type=int, default=0, choices=[0, 1], help="If checkpoint has EMA weights, sample using them.")
    p.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"])

    # --- fallback model config (only used if checkpoint has no payload["config"]) ---
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--y-cont-dim", type=int, default=4)
    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--cond-ch", type=int, default=8)
    p.add_argument("--time-ch", type=int, default=8)

    # --- fallback SDE config ---
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=30.0)

    # output
    p.add_argument("--out-path", default=None, help="Where to save the sample grid png")

    args = p.parse_args()
    device = torch.device(args.device)

    ckpt_path = _infer_ckpt_path(args.out_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")

    # ---- reconstruct model from saved config (preferred) ----
    # Assumes your training code saved something like payload["config"].
    # If not, see fallback below.
    cfg = payload.get("config", None)

    # If the checkpoint didn't save config, fall back to CLI args.
    if cfg is None:
        cfg = {
            "img_ch": 1,
            "n_types": args.n_types,
            "y_cont_dim": args.y_cont_dim,
            "base_ch": args.base_ch,
            "emb_dim": args.emb_dim,
            "cond_ch": args.cond_ch,
            "time_ch": args.time_ch,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
        }

    model = CondUNetTiny(
        n_types=cfg["n_types"],
        y_cont_dim=cfg["y_cont_dim"],
        base_ch=cfg["base_ch"],
        emb_dim=cfg["emb_dim"],
        cond_ch=cfg["cond_ch"],
        time_ch=cfg["time_ch"],
    ).to(device)

    model.load_state_dict(payload["model"])
    if args.use_ema == 1 and ("ema" in payload):
        model.load_state_dict(payload["ema"])
    model.eval()

    sde = VPSDE(
        beta_min=cfg.get("beta_min", 0.1),
        beta_max=cfg.get("beta_max", 30.0),
    )

    if args.out_path is None:
        # save into out_dir/results with informative name
        os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
        args.out_path = os.path.join(
            args.out_dir,
            "results",
            f"samples_ckpt-{os.path.splitext(os.path.basename(ckpt_path))[0]}"
            f"_steps{args.steps}_cfg{args.cfg:.2f}_tend{args.t_end:g}_sampler{args.sampler}_ema{args.use_ema}.png",
        )

    save_sde_samples(
        model=model,
        sde=sde,
        out_path=args.out_path,
        device=device,
        n=args.n,
        theta_max=args.theta_max,
        steps=args.steps,
        cfg=args.cfg,
        t_end=args.t_end,
        sampler=args.sampler
    )

    print(f"Saved samples -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
