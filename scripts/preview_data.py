from __future__ import annotations

import os
import matplotlib.pyplot as plt
from toycrystals.data import ToyCrystalsDataset


def main() -> int:
    os.makedirs("results", exist_ok=True)

    ds = ToyCrystalsDataset(n_samples=10_000, img_size=64, seed=0)

    rows, cols = 6, 6
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        sample = ds[i]
        ax.imshow(sample.x[0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"type={int(sample.y_cat.item())}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    out_path = "results/preview_toycrystals.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
