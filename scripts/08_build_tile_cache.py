"""One-time: raw tiles -> fold-agnostic memmap + metadata + per-fold norm stats.

Run on the PC (tiles live there):
  PPY scripts/08_build_tile_cache.py --limit 50   # smoke test
  PPY scripts/08_build_tile_cache.py              # full build
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from poverty_cnn.data import splits
from poverty_cnn.data.dataset import pad_or_crop_224, N_BANDS, SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/landsat")
    ap.add_argument("--wealth", default="data/processed/wealth_index_clusters.csv")
    ap.add_argument("--out", default="data/processed/tile_cache")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv(args.wealth)
    w["tile"] = ("tile_" + w.country + "_" + w.cluster_id.astype(int).astype(str)
                 + "_" + w.year.astype(int).astype(str) + ".tif")  # int-cast: avoid "1.0"/"2015.0"
    w["path"] = w["tile"].map(lambda t: raw / t)
    w = w[w["path"].map(lambda p: p.exists())].reset_index(drop=True)
    if args.limit:
        w = w.iloc[:args.limit].reset_index(drop=True)
    n = len(w)
    print(f"tiles to cache: {n}")

    cache = np.lib.format.open_memmap(out / "cache.npy", mode="w+",
                                      dtype="float32", shape=(n, N_BANDS, SIZE, SIZE))
    csum = defaultdict(lambda: np.zeros(N_BANDS))
    csq = defaultdict(lambda: np.zeros(N_BANDS))
    ccnt = defaultdict(lambda: np.zeros(N_BANDS))
    nan_fracs = np.zeros(n, "float32")

    for i, row in w.iterrows():
        with rasterio.open(row["path"]) as ds:
            x = ds.read().astype("float32")          # (8,H,W), may contain NaN
        x = pad_or_crop_224(x)
        finite = np.isfinite(x)
        nan_fracs[i] = 1.0 - finite.mean()
        # leakage-free, NaN-aware sufficient stats BEFORE filling
        cc = row["country"]
        xz = np.where(finite, x, 0.0)
        csum[cc] += xz.sum(axis=(1, 2), dtype="float64")
        csq[cc] += (xz.astype("float64") ** 2).sum(axis=(1, 2))
        ccnt[cc] += finite.sum(axis=(1, 2))
        # fill NaN with per-channel finite mean (~0 after z-score)
        with np.errstate(invalid="ignore"):  # all-NaN band -> nan, handled by nan_to_num
            flat = x.reshape(N_BANDS, -1)
            means = np.nanmean(np.where(finite.reshape(N_BANDS, -1), flat, np.nan), axis=1)
        means = np.nan_to_num(means)
        x = np.where(finite, x, means.reshape(N_BANDS, 1, 1)).astype("float32")
        cache[i] = x
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    cache.flush()

    meta = w[["country", "cluster_id", "year", "wealth_index_mean", "urban", "lat", "lon"]].copy()
    meta.insert(0, "row", np.arange(n))
    meta["nan_frac"] = nan_fracs
    meta.to_csv(out / "cache_metadata.csv", index=False)

    # per-fold train normalization from per-country sufficient stats
    stats = {}
    for f in splits.fold_ids():
        tr = splits.countries_for(f, "train")
        S = sum((csum[c] for c in tr), np.zeros(N_BANDS))
        SS = sum((csq[c] for c in tr), np.zeros(N_BANDS))
        Nc = sum((ccnt[c] for c in tr), np.zeros(N_BANDS))
        Nc = np.maximum(Nc, 1)
        mean = S / Nc
        var = np.maximum(SS / Nc - mean ** 2, 1e-12)
        stats[f"{f}_mean"] = mean.astype("float32")
        stats[f"{f}_std"] = np.sqrt(var).astype("float32")
    np.savez(out / "norm_stats.npz", **stats)
    print("wrote", out / "cache.npy", out / "cache_metadata.csv", out / "norm_stats.npz")
    print(f"nan_frac: max={nan_fracs.max()*100:.2f}% mean={nan_fracs.mean()*100:.3f}%")


if __name__ == "__main__":
    main()
