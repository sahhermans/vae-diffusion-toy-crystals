from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from toycrystals.data import ToyCrystalsDataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/toycrystals_train_rotonly.pt")
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--simple", default=False, action="store_true")
    p.add_argument("--rot-only", default=True, action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    ds = ToyCrystalsDataset(n_samples=args.n_samples, img_size=args.img_size, seed=args.seed, n_types=args.n_types, simple=args.simple, rot_only=args.rot_only)

    x_u8 = torch.empty((args.n_samples, 1, args.img_size, args.img_size), dtype=torch.uint8)
    y_cat = torch.empty((args.n_samples,), dtype=torch.int64)
    y_cont = torch.empty((args.n_samples, 4), dtype=torch.float32)

    for i in range(args.n_samples):
        x, yc, yv = ds[i]
        x_u8[i] = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        y_cat[i] = int(yc.item())
        y_cont[i] = yv

        if i % 1000 == 0:
            print(f"{i}/{args.n_samples}")

    torch.save({"x_u8": x_u8, "y_cat": y_cat, "y_cont": y_cont}, out_path)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
