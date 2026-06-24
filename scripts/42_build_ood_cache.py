"""Build the OOD tile cache (memmap + metadata) from the downloaded OOD tiles.

Mirrors scripts/08 exactly (pad/crop to 224, NaN-aware per-channel mean fill) so OOD tiles
are processed identically to training. Labels = the frozen-PCA wealth (multiround axis). No
norm stats here — at test time we normalize with each MODEL's OWN training norm stats (frozen).

Output: data/processed/tile_cache_ood/{cache.npy, cache_metadata.csv}
Run on PC: PPY scripts/42_build_ood_cache.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from poverty_cnn.data.dataset import pad_or_crop_224, N_BANDS, SIZE

RAW = Path("data/raw/landsat_ood")
TARGETS = "data/processed/ood_extract_targets.csv"
OUT = Path("data/processed/tile_cache_ood")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(TARGETS)
    t["tile"] = ("tile_" + t.country + "_" + t.cluster_id.astype(int).astype(str)
                 + "_" + t.survey_year.astype(int).astype(str) + ".tif")
    t["path"] = t.tile.map(lambda x: RAW / x)
    t = t[t.path.map(lambda p: p.exists())].reset_index(drop=True)
    n = len(t)
    print(f"OOD tiles found on disk: {n} / {len(pd.read_csv(TARGETS))} targets")

    cache = np.lib.format.open_memmap(OUT / "cache.npy", mode="w+",
                                      dtype="float32", shape=(n, N_BANDS, SIZE, SIZE))
    nan_fracs = np.zeros(n, "float32")
    for i, row in t.iterrows():
        with rasterio.open(row.path) as ds:
            x = ds.read().astype("float32")
        x = pad_or_crop_224(x)
        finite = np.isfinite(x)
        nan_fracs[i] = 1.0 - finite.mean()
        with np.errstate(invalid="ignore"):
            means = np.nanmean(np.where(finite.reshape(N_BANDS, -1), x.reshape(N_BANDS, -1), np.nan), axis=1)
        means = np.nan_to_num(means)
        x = np.where(finite, x, means.reshape(N_BANDS, 1, 1)).astype("float32")
        cache[i] = x
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)
    cache.flush()

    meta = t[["country", "cluster_id", "survey_year", "urban", "wealth_index_mean", "lat", "lon"]].copy()
    meta = meta.rename(columns={"survey_year": "year"})
    meta.insert(0, "row", np.arange(n))
    meta["nan_frac"] = nan_fracs
    meta.to_csv(OUT / "cache_metadata.csv", index=False)
    print(f"wrote OOD cache ({n} tiles) + metadata | nan max {nan_fracs.max()*100:.1f}% mean {nan_fracs.mean()*100:.2f}%")
    print("per-country:", meta.groupby("country").size().to_dict())


if __name__ == "__main__":
    main()
