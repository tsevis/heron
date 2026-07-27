"""Datasets for visible<->thermal translation training (CLAUDE.md §2.2B).

The candidate public corpora: **KAIST Multispectral Pedestrian**, **FLIR ADAS**,
Iris/Eurecom thermal faces, AVIID and DroneVehicle. This module does not download
anything by itself — several require registration/licence acceptance — it defines
the loaders once the data is on disk.

Layout expected::

    data/thermal/
      paired/   trainA/xxx.png   trainB/xxx.png   # aligned pairs, same filename
      unpaired/ trainA/*.png     trainB/*.png     # A = visible, B = thermal
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SOURCES = {
    "kaist": "https://soonminhwang.github.io/rgbt-ped-detection/  (paired RGB/LWIR pedestrians)",
    "flir_adas": "https://www.flir.com/oem/adas/adas-dataset-form/  (registration required)",
    "eurecom_faces": "http://vis-www.cs.umass.edu/~..  Iris/Eurecom thermal face sets",
    "avid_drone": "DroneVehicle / AVIID aerial RGB-IR sets",
}


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXT)


def _load(path: Path, size: int) -> torch.Tensor:
    """Load an image as a [-1,1] CHW tensor (the GAN convention)."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


class UnpairedThermalDataset(Dataset):
    """For CycleGAN: independent visible (A) and thermal (B) collections."""

    def __init__(self, root: str | Path = "data/thermal/unpaired", size: int = 256, seed: int = 0):
        root = Path(root)
        self.a = _list_images(root / "trainA")
        self.b = _list_images(root / "trainB")
        self.size = size
        self.rng = random.Random(seed)
        if not self.a or not self.b:
            raise FileNotFoundError(
                f"no images under {root}/trainA and {root}/trainB. "
                f"Fetch a corpus first — candidates: {list(SOURCES)}"
            )

    def __len__(self) -> int:
        return max(len(self.a), len(self.b))

    def __getitem__(self, i: int):
        a = self.a[i % len(self.a)]
        b = self.b[self.rng.randrange(len(self.b))]   # unpaired: random partner
        return _load(a, self.size), _load(b, self.size)


class PairedThermalDataset(Dataset):
    """For Pix2PixHD: pixel-aligned pairs sharing a filename across trainA/trainB."""

    def __init__(self, root: str | Path = "data/thermal/paired", size: int = 512):
        root = Path(root)
        a_dir, b_dir = root / "trainA", root / "trainB"
        self.pairs = [(p, b_dir / p.name) for p in _list_images(a_dir) if (b_dir / p.name).exists()]
        self.size = size
        if not self.pairs:
            raise FileNotFoundError(
                f"no aligned pairs under {root} (trainA/trainB must share filenames). "
                f"Candidates: {list(SOURCES)}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        a, b = self.pairs[i]
        return _load(a, self.size), _load(b, self.size)


def describe() -> str:
    lines = ["Visible<->thermal corpora (download manually, licences vary):"]
    lines += [f"  {k:16s} {v}" for k, v in SOURCES.items()]
    return "\n".join(lines)
