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


def test_make_fold_loaders_roundtrip(tmp_path):
    import pandas as pd
    from poverty_cnn.data.dataset import make_fold_loaders
    from poverty_cnn.data import splits
    # synthetic cache dir: 23 countries x 2 rows each
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
    counts = {k: len(loaders[k].dataset) for k in loaders}
    assert sum(counts.values()) == n   # disjoint train/val/test cover all rows


def test_channel_subset():
    cache, meta, mean, std = _toy()
    ds7 = PovertyTileDataset(cache, meta, np.arange(len(meta)), mean, std,
                             channels=[0, 1, 2, 3, 4, 5, 6])   # MS-only
    ds1 = PovertyTileDataset(cache, meta, np.arange(len(meta)), mean, std,
                             channels=[7])                       # NL-only
    assert ds7[0][0].shape == (7, 224, 224)
    assert ds1[0][0].shape == (1, 224, 224)
    assert ds7[0][0].dtype == torch.float32
