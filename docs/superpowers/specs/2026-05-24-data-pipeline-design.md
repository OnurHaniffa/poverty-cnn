# Data Pipeline Design — tiles + wealth → trainable PyTorch dataset

**Date:** 2026-05-24
**Author:** Onur Haniffa (with Claude)
**Scope:** `splits.py`, a one-time tile-cache builder, and `dataset.py`. This is the
bridge between the extracted data (8-channel Landsat+NL tiles in `data/raw/landsat/`,
wealth index in `data/processed/wealth_index_clusters.csv`) and model training.
Model architecture and the training loop are **out of scope** (separate spec).

## Goal

Produce a `torch.utils.data.Dataset` that yields `(image[8,224,224], wealth, meta)`
pairs, partitioned by Yeh 2020's exact 5-fold cross-country protocol, with
leakage-free per-channel normalization — fast enough to keep 2× P5000 GPUs fed.

## Ground truth (verified 2026-05-24)

- Tiles: 8-band float32 GeoTIFF. **Dimensions vary widely: 223×225 … 260×226**
  (not a fixed 227×226) — degree-grid approximation varies with latitude. **101
  tiles are <224 px in width** (223), so a plain center-crop-to-224 is impossible
  on those → must pad-then-crop. Per-band scales differ wildly (optical ~7k–26k
  scaled-SR ints, thermal ~47k, nightlights ~5–117). **NaN ranges 0–~12%** per
  tile (median-composite gaps; sample mean ~1.2%), not the ~0.07% one sample showed.
- Wealth table columns: `country, cluster_id, wealth_index_mean, n_households,
  urban, lat, lon, year`. `cluster_id` is the bare DHS cluster integer.
- **Join key:** `tile_{country}_{cluster_id}_{year}.tif`.
- 13,453 of 13,472 tiles exist; the 19 missing are DHS `SOURCE=="MIS"` null-island
  `(0,0)` clusters — unrecoverable, correctly excluded.
- Fold assignment available verbatim from `wilds` 2.0.0
  `wilds.datasets.poverty_dataset._SURVEY_NAMES_2009_17{A..E}` (== Yeh Supp Table S2).
  `DHS_COUNTRIES` matches our 23 countries exactly.

## Components

### 1. `src/poverty_cnn/data/splits.py`
Vendors Yeh's 5 folds (no runtime `wilds` dependency — copied in with a source
comment, so the partition is explicit and reviewable).

- `COUNTRY_CODE_TO_NAME: dict[str, str]` — `"AO" → "angola"`, … (our 23 codes →
  the WILDS country-name strings). Single source of truth for the code↔name map.
- `FOLDS: dict[str, dict[str, list[str]]]` — keys `"A".."E"`; each value
  `{"train": [...], "val": [...], "test": [...]}` of **country names**, copied
  from `_SURVEY_NAMES_2009_17{A..E}`.
- `clusters_for(metadata: pd.DataFrame, fold: str, role: str) -> pd.Index` —
  returns the row indices of `metadata` whose country falls in `FOLDS[fold][role]`.
- Sanity asserts at import: every country appears in `test` exactly once across
  folds; train/val/test are disjoint within a fold; all 23 codes map to a name.

### 2. `scripts/08_build_tile_cache.py` (one-time preprocessing)
Runs on the PC after the download completes. Produces fold-agnostic RAW cache.

- Load wealth table; build `tile_path` per row; **keep rows whose tile exists on
  disk** (this is the trainability filter — auto-drops the 19 + any orphan).
- Allocate memmap `cache.npy`, shape `[N, 8, 224, 224]`, float32 (~22 GB).
- For each kept row, in stable sorted order:
  - `rasterio` read → `(8, H, W)`.
  - **Pad-then-center-crop to exactly 224×224**: reflect-pad any dim <224 up to
    224, then center-crop. Handles the 223-wide runts and the 260-wide giants
    uniformly; the fixed 224 normalizes every footprint to ~6.72 km.
  - Record `nan_frac` (fraction NaN before fill), then **NaN → per-channel mean**
    of that tile (≈0 after later z-score).
  - Write into `cache[i]`.
- Save `cache_metadata.csv` aligned row-for-row with the memmap: `row, country,
  cluster_id, year, wealth_index_mean, urban, lat, lon, nan_frac`.
- **Compute per-fold normalization** and save `norm_stats.npz`: for each fold,
  per-channel `mean[8]`, `std[8]` computed over **that fold's `train` clusters'
  pixels only** (running sums over the memmap rows for those clusters).
- Idempotent / resumable: skip if outputs exist and row count matches; `--force`
  rebuilds.

Artifacts (`cache.npy`, `cache_metadata.csv`, `norm_stats.npz`) live under
`data/processed/tile_cache/` — **gitignored** (data, not code).

### 3. `src/poverty_cnn/data/dataset.py`
- `PovertyTileDataset(Dataset)`:
  - `__init__(memmap, metadata, row_indices, mean, std, augment: bool)` —
    `row_indices` selects this split's rows (from `splits.clusters_for`);
    `mean`/`std` are the active fold's **train** stats.
  - `__getitem__(i)`: read `memmap[row]` (zero decode) → `(x - mean)/std`
    per channel → if `augment`: random h/v flip + random 0/90/180/270° rotation
    (label-preserving for nadir imagery) → return
    `(FloatTensor[8,224,224], FloatTensor(wealth), meta_dict)`.
  - `__len__`.
- A `make_fold_loaders(fold, batch_size, ...)` helper wires memmap + metadata +
  `splits.clusters_for` + `norm_stats` into train/val/test `DataLoader`s. Train
  loader: `shuffle=True`, augment on, `num_workers>0`, `pin_memory=True`. Val/test:
  no shuffle, no augment, train-fold stats.

## Data flow

```
wealth_index_clusters.csv ─┐
data/raw/landsat/*.tif ─────┼─► 08_build_tile_cache.py ─► cache.npy (memmap, raw)
                            │                              cache_metadata.csv
                            │                              norm_stats.npz (per-fold)
                            └─────────────────────────────────────────────┘
                                                                 │
splits.py (Yeh folds) ──► clusters_for(meta, fold, role) ──► row_indices
                                                                 │
            cache.npy + metadata + row_indices + fold stats ─► PovertyTileDataset ─► DataLoader ─► model
```

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Tile store | single ~22 GB float32 **memmap**, raw (un-normalized) | zero-decode random access keeps GPUs fed; raw → fold-agnostic |
| Normalization | **per-fold train** per-channel mean/std | leakage-free; test-country pixel stats never touch training |
| Crop | reflect-pad any dim <224 → then center-crop to 224 | dims vary 223–260; pad-then-crop is uniform; fixed 224 normalizes footprint to ~6.72 km |
| NaN | record `nan_frac`, fill with per-channel tile mean | 0–12% per tile; fill ≈0 post z-score; nan_frac kept for QC |
| Augment | flips + 90° rotations, train only | label-preserving for top-down imagery |
| Trainability | tile-file-exists ∩ has-wealth | auto-excludes null-island 19 + orphans |
| Fold source | vendored from WILDS `_SURVEY_NAMES` | == Yeh Supp Table S2, apples-to-apples |

## Testing

- `splits.py`: unit-test the import-time invariants (each country tested once;
  disjoint roles; full code→name coverage).
- Cache builder: dry-run against the ~hundreds of tiles already downloaded
  (build a tiny cache, assert shape `[n,8,224,224]`, no NaN, finite stats).
- `dataset.py`: assert `__getitem__` returns `[8,224,224]`, per-channel mean≈0 /
  std≈1 on a train batch under that fold's stats, augmentation changes pixels but
  not the label, val/test loaders apply train-fold stats and don't augment.

## Sequencing / dependencies

- Code (`splits.py`, `dataset.py`, builder) can be **written and tested now**
  against the 580+ tiles already on disk.
- Full `cache.npy` build waits for the rclone download to finish (~overnight).
- Next spec: model (`poverty_resnet.py`) + training loop.

## Out of scope

Model architecture, training loop, hyperparameter search, eval/fairness — later specs.
