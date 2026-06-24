# Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn 13,453 downloaded 8-channel Landsat+NL tiles + the wealth index into a fold-partitioned, leakage-free PyTorch dataset ready for training.

**Architecture:** A one-time builder converts raw GeoTIFFs into a single fold-agnostic float32 memmap (`cache.npy`) + row-aligned `metadata.csv` + per-fold train normalization stats (`norm_stats.npz`). `splits.py` vendors Yeh 2020's exact 5-fold country partition (from WILDS). `dataset.py` reads memmap rows with zero decode, applies the active fold's z-score + train-only augmentation.

**Tech Stack:** Python 3.11, NumPy, pandas, rasterio, PyTorch 2.2, pytest. Source spec: `docs/superpowers/specs/2026-05-24-data-pipeline-design.md`.

**Where things run:**
- Tasks 1–3 (pure logic / synthetic data): **Mac** conda python `/opt/homebrew/Caskroom/miniforge/base/envs/poverty-cnn/bin/python` (alias below: `MPY`). Fast TDD.
- Tasks 4–5 (need the real tiles): **lab PC** via `ssh sim`, python `~/miniconda3/envs/poverty-cnn/bin/python` (`PPY`). Code reaches the PC via `rsync` (Task 4, Step 1).

```
MPY=/opt/homebrew/Caskroom/miniforge/base/envs/poverty-cnn/bin/python
PPY=~/miniconda3/envs/poverty-cnn/bin/python   # used inside `ssh sim '...'`
```

**Commit gate:** The user commits only on explicit go-ahead ("commit it"). Commit steps below are the natural checkpoints — during execution, *stage* and pause at each; squash/commit to a branch once the user approves.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/poverty_cnn/data/splits.py` (new) | Yeh's 5-fold country partition + code↔name map + `clusters_for()` |
| `src/poverty_cnn/data/dataset.py` (new) | `pad_or_crop_224`, `PovertyTileDataset`, `make_fold_loaders` |
| `scripts/08_build_tile_cache.py` (new) | one-time: tiles → memmap + metadata + per-fold norm stats |
| `tests/data/test_splits.py` (new) | fold invariants + `clusters_for` |
| `tests/data/test_dataset.py` (new) | crop/pad, normalization, augmentation, loaders (synthetic memmap) |

---

## Task 1: `splits.py` — Yeh's 5-fold partition

**Files:**
- Create: `src/poverty_cnn/data/splits.py`
- Test: `tests/data/test_splits.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_splits.py
import numpy as np
import pandas as pd
import pytest
from poverty_cnn.data import splits


def test_all_23_countries_mapped():
    assert len(splits.COUNTRY_CODE_TO_NAME) == 23
    assert len(set(splits.COUNTRY_CODE_TO_NAME.values())) == 23


def test_each_country_tested_exactly_once():
    tested = [c for f in splits.fold_ids() for c in splits.countries_for(f, "test")]
    assert sorted(tested) == sorted(splits.COUNTRY_CODE_TO_NAME)  # 23, no repeats


def test_roles_disjoint_and_cover_within_fold():
    for f in splits.fold_ids():
        tr = splits.countries_for(f, "train")
        va = splits.countries_for(f, "val")
        te = splits.countries_for(f, "test")
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert tr | va | te == set(splits.COUNTRY_CODE_TO_NAME)


def test_clusters_for_returns_row_positions():
    meta = pd.DataFrame({"country": ["AO", "KE", "AO", "ZW"]})
    # fold A tests angola(AO); KE/ZW are not in A-test
    rows = splits.clusters_for(meta, "A", "test")
    assert set(rows.tolist()) == {0, 2}


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        splits.countries_for("A", "bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPY -m pytest tests/data/test_splits.py -q`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (splits not implemented).

- [ ] **Step 3: Write minimal implementation**

```python
# src/poverty_cnn/data/splits.py
"""Yeh 2020 5-fold cross-country split (== WILDS PovertyMap _SURVEY_NAMES_2009_17).

Vendored verbatim so the partition is explicit and reviewable, with no runtime
`wilds` dependency. Source: wilds 2.0.0 wilds/datasets/poverty_dataset.py.
Each country is in `test` exactly once across the 5 folds (apples-to-apples with
Yeh 2020 Supplementary Table S2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Our 23 DHS country codes -> WILDS country-name strings.
COUNTRY_CODE_TO_NAME = {
    "AO": "angola", "BJ": "benin", "BF": "burkina_faso", "CM": "cameroon",
    "CI": "cote_d_ivoire", "CD": "democratic_republic_of_congo", "ET": "ethiopia",
    "GH": "ghana", "GN": "guinea", "KE": "kenya", "LS": "lesotho", "MW": "malawi",
    "ML": "mali", "MZ": "mozambique", "NG": "nigeria", "RW": "rwanda",
    "SN": "senegal", "SL": "sierra_leone", "TZ": "tanzania", "TG": "togo",
    "UG": "uganda", "ZM": "zambia", "ZW": "zimbabwe",
}
NAME_TO_CODE = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}

# test + val country names per fold (train = all - test - val). From WILDS.
_FOLDS_NAMES = {
    "A": {"test": ["angola", "cote_d_ivoire", "ethiopia", "mali", "rwanda"],
          "val":  ["benin", "burkina_faso", "guinea", "sierra_leone", "tanzania"]},
    "B": {"test": ["benin", "burkina_faso", "guinea", "sierra_leone", "tanzania"],
          "val":  ["cameroon", "ghana", "malawi", "zimbabwe"]},
    "C": {"test": ["cameroon", "ghana", "malawi", "zimbabwe"],
          "val":  ["democratic_republic_of_congo", "mozambique", "nigeria", "togo", "uganda"]},
    "D": {"test": ["democratic_republic_of_congo", "mozambique", "nigeria", "togo", "uganda"],
          "val":  ["kenya", "lesotho", "senegal", "zambia"]},
    "E": {"test": ["kenya", "lesotho", "senegal", "zambia"],
          "val":  ["angola", "cote_d_ivoire", "ethiopia", "mali", "rwanda"]},
}


def fold_ids() -> list[str]:
    return list("ABCDE")


def countries_for(fold: str, role: str) -> set[str]:
    """Return the set of country CODES for a fold/role."""
    f = _FOLDS_NAMES[fold]
    test = {NAME_TO_CODE[n] for n in f["test"]}
    val = {NAME_TO_CODE[n] for n in f["val"]}
    if role == "test":
        return test
    if role == "val":
        return val
    if role == "train":
        return set(COUNTRY_CODE_TO_NAME) - test - val
    raise ValueError(f"role must be train/val/test, got {role!r}")


def clusters_for(metadata: pd.DataFrame, fold: str, role: str) -> np.ndarray:
    """Integer row positions of `metadata` whose `country` belongs to fold/role."""
    codes = countries_for(fold, role)
    return np.flatnonzero(metadata["country"].isin(codes).to_numpy())


# --- import-time invariants (fail fast on a bad edit) ---
_tested = [c for f in fold_ids() for c in countries_for(f, "test")]
assert sorted(_tested) == sorted(COUNTRY_CODE_TO_NAME), "each country must be tested exactly once"
for _f in fold_ids():
    _tr, _va, _te = (countries_for(_f, r) for r in ("train", "val", "test"))
    assert _tr.isdisjoint(_va) and _tr.isdisjoint(_te) and _va.isdisjoint(_te)
    assert _tr | _va | _te == set(COUNTRY_CODE_TO_NAME)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPY -m pytest tests/data/test_splits.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Stage + checkpoint**

```bash
git add src/poverty_cnn/data/splits.py tests/data/test_splits.py
# commit when user says "commit it":  git commit -m "feat(data): vendor Yeh 2020 5-fold cross-country splits"
```

---

## Task 2: `dataset.py` — crop/pad + `PovertyTileDataset`

**Files:**
- Create: `src/poverty_cnn/data/dataset.py`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py
import random

import numpy as np
import torch
from poverty_cnn.data.dataset import pad_or_crop_224, PovertyTileDataset


def test_pad_or_crop_handles_oversize_and_undersize():
    big = np.random.rand(8, 260, 226).astype("float32")
    small = np.random.rand(8, 223, 225).astype("float32")
    assert pad_or_crop_224(big).shape == (8, 224, 224)
    assert pad_or_crop_224(small).shape == (8, 224, 224)


def _toy(n=6):
    cache = np.random.rand(n, 8, 224, 224).astype("float32") * 1000
    import pandas as pd
    meta = pd.DataFrame({
        "country": ["AO", "KE", "AO", "ZW", "KE", "AO"][:n],
        "wealth_index_mean": np.linspace(-1, 1, n).astype("float32"),
    })
    mean = cache.mean(axis=(0, 2, 3)); std = cache.std(axis=(0, 2, 3))
    return cache, meta, mean, std


def test_getitem_shapes_and_normalization():
    cache, meta, mean, std = _toy()
    ds = PovertyTileDataset(cache, meta, np.arange(len(meta)), mean, std, augment=False)
    x, y, m = ds[0]
    assert x.shape == (8, 224, 224) and x.dtype == torch.float32
    assert isinstance(y.item(), float)
    # whole-set per-channel mean ~0 after z-score
    allx = torch.stack([ds[i][0] for i in range(len(ds))])
    assert allx.mean().abs().item() < 0.1


def test_augmentation_changes_pixels_not_label():
    cache, meta, mean, std = _toy()
    ds = PovertyTileDataset(cache, meta, np.arange(len(meta)), mean, std, augment=True)
    base = (cache[0] - mean.reshape(8, 1, 1)) / std.reshape(8, 1, 1)
    label = float(meta.iloc[0]["wealth_index_mean"])
    seen_diff = False
    for seed in range(10):
        random.seed(seed)
        x, y, _ = ds[0]
        assert x.shape == (8, 224, 224)
        assert abs(y.item() - label) < 1e-6           # label invariant to augmentation
        if not np.allclose(x.numpy(), base):
            seen_diff = True
    assert seen_diff                                   # augmentation perturbs pixels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPY -m pytest tests/data/test_dataset.py -q`
Expected: FAIL — `ImportError` (dataset not implemented).

- [ ] **Step 3: Write minimal implementation**

```python
# src/poverty_cnn/data/dataset.py
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
                 augment: bool = False):
        self.cache = cache
        self.meta = metadata
        self.rows = np.asarray(row_indices)
        self.mean = np.asarray(mean, dtype="float32").reshape(N_BANDS, 1, 1)
        self.std = np.asarray(std, dtype="float32").reshape(N_BANDS, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        x = np.asarray(self.cache[r], dtype="float32")
        x = (x - self.mean) / self.std
        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, random.randint(0, 3), axes=(1, 2))
            x = np.ascontiguousarray(x)
        y = np.float32(self.meta.iloc[r]["wealth_index_mean"])
        row = self.meta.iloc[r]
        meta = {"country": row["country"], "row": r}
        return torch.from_numpy(x), torch.tensor(y), meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPY -m pytest tests/data/test_dataset.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Stage + checkpoint**

```bash
git add src/poverty_cnn/data/dataset.py tests/data/test_dataset.py
# commit when approved: git commit -m "feat(data): PovertyTileDataset + pad_or_crop_224"
```

---

## Task 3: `make_fold_loaders` — wire memmap + splits + stats

**Files:**
- Modify: `src/poverty_cnn/data/dataset.py` (append)
- Test: `tests/data/test_dataset.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/test_dataset.py
def test_make_fold_loaders_roundtrip(tmp_path):
    import pandas as pd
    from poverty_cnn.data.dataset import make_fold_loaders
    # synthetic cache dir: 23 countries x 2 rows each
    from poverty_cnn.data import splits
    codes = list(splits.COUNTRY_CODE_TO_NAME)
    rows_country = [c for c in codes for _ in range(2)]
    n = len(rows_country)
    cache = np.random.rand(n, 8, 224, 224).astype("float32")
    np.save(tmp_path / "cache.npy", cache)
    pd.DataFrame({"country": rows_country,
                  "wealth_index_mean": np.zeros(n, "float32")}).to_csv(
        tmp_path / "cache_metadata.csv", index=False)
    stats = {}
    for f in splits.fold_ids():
        stats[f"{f}_mean"] = cache.mean(axis=(0, 2, 3))
        stats[f"{f}_std"] = cache.std(axis=(0, 2, 3))
    np.savez(tmp_path / "norm_stats.npz", **stats)

    loaders = make_fold_loaders(tmp_path, "A", batch_size=4, num_workers=0)
    assert set(loaders) == {"train", "val", "test"}
    xb, yb, mb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (8, 224, 224)
    # train/val/test row counts sum to n, disjoint
    counts = {k: len(loaders[k].dataset) for k in loaders}
    assert sum(counts.values()) == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPY -m pytest tests/data/test_dataset.py::test_make_fold_loaders_roundtrip -q`
Expected: FAIL — `ImportError: cannot import name 'make_fold_loaders'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/poverty_cnn/data/dataset.py
def make_fold_loaders(cache_dir, fold: str, batch_size: int = 64,
                      num_workers: int = 4):
    """Build train/val/test DataLoaders for one fold from a built cache dir.

    cache_dir must contain cache.npy, cache_metadata.csv, norm_stats.npz.
    Normalization uses the fold's TRAIN stats for all three splits.
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
        ds = PovertyTileDataset(cache, meta, rows, mean, std, augment=aug)
        loaders[role] = DataLoader(ds, batch_size=batch_size, shuffle=shuf,
                                   num_workers=num_workers, pin_memory=True)
    return loaders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPY -m pytest tests/data/test_dataset.py -q`
Expected: PASS (all dataset tests).

- [ ] **Step 5: Stage + checkpoint**

```bash
git add src/poverty_cnn/data/dataset.py tests/data/test_dataset.py
# commit when approved: git commit -m "feat(data): make_fold_loaders"
```

---

## Task 4: `08_build_tile_cache.py` — builder + PC smoke test

**Files:**
- Create: `scripts/08_build_tile_cache.py`

- [ ] **Step 1: Write the builder**

```python
# scripts/08_build_tile_cache.py
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
```

- [ ] **Step 2: Sync code to the PC**

Run:
```bash
rsync -av --exclude='__pycache__' \
  /Users/onurmohamedhaniffa/poverty-cnn/src/poverty_cnn/data/ \
  sim:poverty-cnn/src/poverty_cnn/data/
rsync -av /Users/onurmohamedhaniffa/poverty-cnn/scripts/08_build_tile_cache.py \
  sim:poverty-cnn/scripts/08_build_tile_cache.py
```
Expected: `splits.py`, `dataset.py`, `08_build_tile_cache.py` transferred.

- [ ] **Step 3: Smoke-test the builder on 50 tiles (PC)**

Run:
```bash
ssh sim 'cd ~/poverty-cnn && ~/miniconda3/envs/poverty-cnn/bin/python scripts/08_build_tile_cache.py --limit 50 --out data/processed/tile_cache_smoke --force'
```
Expected: prints `tiles to cache: 50`, then the three artifact paths, then a `nan_frac` line. No traceback.

- [ ] **Step 4: Verify smoke artifacts (PC)**

Run:
```bash
ssh sim 'cd ~/poverty-cnn && ~/miniconda3/envs/poverty-cnn/bin/python - <<PY
import numpy as np, pandas as pd
c = np.load("data/processed/tile_cache_smoke/cache.npy", mmap_mode="r")
m = pd.read_csv("data/processed/tile_cache_smoke/cache_metadata.csv")
s = np.load("data/processed/tile_cache_smoke/norm_stats.npz")
assert c.shape == (50, 8, 224, 224), c.shape
assert len(m) == 50 and list(m.columns)[0] == "row"
assert np.isfinite(c[:5]).all(), "NaNs leaked into cache"
assert s["A_mean"].shape == (8,) and np.isfinite(s["A_std"]).all()
print("SMOKE OK", c.shape, "| A_mean", np.round(s["A_mean"],1))
PY'
```
Expected: `SMOKE OK (50, 8, 224, 224) | A_mean [...]`.

- [ ] **Step 5: Stage + checkpoint**

```bash
git add scripts/08_build_tile_cache.py
# commit when approved: git commit -m "feat(data): tile cache builder (memmap + per-fold norm stats)"
```

---

## Task 5: Full cache build + end-to-end integration (PC)

**Files:** none new — runs the pipeline on all 13,453 tiles.

- [ ] **Step 1: Full build (PC)**

Run:
```bash
ssh sim 'cd ~/poverty-cnn && ~/miniconda3/envs/poverty-cnn/bin/python scripts/08_build_tile_cache.py --out data/processed/tile_cache --force'
```
Expected: `tiles to cache: 13453`, progress every 500, final artifact + nan_frac line. ~22 GB `cache.npy`.

- [ ] **Step 2: Integration — load real cache through `make_fold_loaders` (PC)**

Run:
```bash
ssh sim 'cd ~/poverty-cnn && ~/miniconda3/envs/poverty-cnn/bin/python - <<PY
import numpy as np, torch
from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.data import splits
L = make_fold_loaders("data/processed/tile_cache", "A", batch_size=64, num_workers=4)
n = {k: len(L[k].dataset) for k in L}
print("split sizes:", n, "total", sum(n.values()))
xb, yb, mb = next(iter(L["train"]))
print("batch:", tuple(xb.shape), "y", tuple(yb.shape))
# train batch should be ~z-scored under fold-A train stats
print("batch per-chan mean ~0:", float(xb.mean()), "std ~1:", float(xb.std()))
# test countries for fold A are the held-out 5
print("A test countries:", sorted(splits.countries_for("A","test")))
PY'
```
Expected: total == 13453; batch `(64, 8, 224, 224)`; mean ≈ 0, std ≈ 1 (±0.2); `A test countries: ['AO','CI','ET','ML','RW']`.

- [ ] **Step 3: QC the NaN distribution (PC)**

Run:
```bash
ssh sim 'cd ~/poverty-cnn && ~/miniconda3/envs/poverty-cnn/bin/python - <<PY
import pandas as pd
m = pd.read_csv("data/processed/tile_cache/cache_metadata.csv")
print("rows:", len(m), "| countries:", m.country.nunique())
print("nan_frac: max %.2f%% mean %.3f%%" % (m.nan_frac.max()*100, m.nan_frac.mean()*100))
print("tiles >5%% NaN:", int((m.nan_frac>0.05).sum()))
print(m.country.value_counts().sort_index().to_string())
PY'
```
Expected: 13453 rows, 23 countries, per-country counts matching the verified set (AO 625 … ZW 400). Note how many tiles exceed 5% NaN (QC flag for later, not a blocker).

- [ ] **Step 4: Stage + checkpoint**

```bash
# tile_cache/ is data — gitignored, not committed. Nothing to add here.
echo "pipeline-v1 data ready"
# checkpoint commit (code only) when approved: git commit -m "chore: data pipeline v1 built + verified"
```

---

## Self-Review (completed)

- **Spec coverage:** splits (Task 1), memmap + metadata + per-fold stats builder (Task 4), pad-then-crop + NaN-record + per-fold z-score + augmentation + loaders (Tasks 2–3), full build + integration + QC (Task 5). Trainability filter = `path.exists()` (Task 4 Step 1). All spec sections mapped.
- **Placeholders:** none — every code/command step is concrete.
- **Type consistency:** `pad_or_crop_224`, `PovertyTileDataset(cache, metadata, row_indices, mean, std, augment)`, `make_fold_loaders(cache_dir, fold, batch_size, num_workers)`, `splits.clusters_for(metadata, fold, role)`, `splits.countries_for(fold, role)`, `fold_ids()` — names/signatures consistent across tasks and the builder. Cache artifacts (`cache.npy`, `cache_metadata.csv`, `norm_stats.npz`) and stat keys (`{fold}_mean`/`{fold}_std`) consistent between Task 4 (writer) and Task 3 (reader).
