"""PyTorch dataset over the pre-built tile memmap."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from poverty_cnn.data import splits

SIZE = 224
N_BANDS = 8


def pad_or_crop_224(arr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Reflect-pad any spatial dim < size up to size, then center-crop to size."""
    _, h, w = arr.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        arr = np.pad(arr, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                     mode="reflect")
    _, h, w = arr.shape
    top, left = (h - size) // 2, (w - size) // 2
    return arr[:, top:top + size, left:left + size]


class PovertyTileDataset(Dataset):
    """Yields (image[8,224,224] z-scored, wealth, meta) from a fold-agnostic memmap.

    cache: array-like [N,8,224,224] raw float32 (memmap ok).
    metadata: DataFrame row-aligned to cache (columns incl. country, wealth_index_mean).
    row_indices: positions into cache/metadata for this split.
    mean/std: (8,) per-channel stats of the ACTIVE fold's train set.
    """

    def __init__(self, cache, metadata: pd.DataFrame, row_indices, mean, std,
                 augment: bool = False, channels=None):
        self.cache = cache
        self.rows = np.asarray(row_indices)
        # pre-extract to NumPy so __getitem__ never touches pandas (hot path)
        self.wealth = metadata["wealth_index_mean"].to_numpy(dtype="float32")
        self.countries = metadata["country"].to_numpy()
        # channel subset (for MS-only / NL-only ablations); None = all 8 bands
        self.channels = list(range(N_BANDS)) if channels is None else list(channels)
        nc = len(self.channels)
        self.mean = np.asarray(mean, dtype="float32")[self.channels].reshape(nc, 1, 1)
        self.std = np.asarray(std, dtype="float32")[self.channels].reshape(nc, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        x = np.asarray(self.cache[r], dtype="float32")[self.channels]
        x = (x - self.mean) / self.std
        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, random.randint(0, 3), axes=(1, 2))
            x = np.ascontiguousarray(x)
        y = self.wealth[r]
        meta = {"country": str(self.countries[r]), "row": r}
        return torch.from_numpy(x), torch.tensor(y), meta


def make_fold_loaders(cache_dir, fold: str, batch_size: int = 64,
                      num_workers: int = 4, channels=None):
    """Build train/val/test DataLoaders for one fold from a built cache dir.

    cache_dir must contain cache.npy, cache_metadata.csv, norm_stats.npz.
    Normalization uses the fold's TRAIN stats for all three splits.
    `channels` selects a band subset (e.g. [0..6] MS-only, [7] NL-only); None = all 8.
    """
    cache_dir = Path(cache_dir)
    cache = np.load(cache_dir / "cache.npy", mmap_mode="r")
    meta = pd.read_csv(cache_dir / "cache_metadata.csv")
    stats = np.load(cache_dir / "norm_stats.npz")
    mean, std = stats[f"{fold}_mean"], stats[f"{fold}_std"]

    loaders = {}
    for role, aug, shuf in [("train", True, True), ("val", False, False),
                            ("test", False, False)]:
        rows = splits.clusters_for(meta, fold, role)
        ds = PovertyTileDataset(cache, meta, rows, mean, std, augment=aug,
                                channels=channels)
        loaders[role] = DataLoader(ds, batch_size=batch_size, shuffle=shuf,
                                   num_workers=num_workers, pin_memory=True)
    return loaders
