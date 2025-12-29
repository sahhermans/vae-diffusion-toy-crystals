from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


class ToyCrystalsDiskDataset(Dataset):
    """
    Loads a precomputed dataset saved by scripts/build_dataset.py.

    Stored images are uint8 in [0,255] to keep files small.
    Returned images are float32 in [0,1].
    """

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        obj = torch.load(p, map_location="cpu")

        self.x_u8: torch.Tensor = obj["x_u8"]      # [N,1,H,W] uint8
        self.y_cat: torch.Tensor = obj["y_cat"]    # [N] int64
        self.y_cont: torch.Tensor = obj["y_cont"]  # [N,4] float32

    def __len__(self) -> int:
        return int(self.x_u8.shape[0])

    def __getitem__(self, idx: int):
        x = self.x_u8[idx].to(torch.float32) / 255.0
        y_cat = self.y_cat[idx]
        y_cont = self.y_cont[idx]
        return x, y_cat, y_cont
