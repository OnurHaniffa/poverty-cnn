"""OOD channel ablation ('feature-out'): which bands drive the OOD predictions, and is the
South Africa break band-specific?

Zero each band-group (after normalisation = replace with its mean) and re-run the frozen 36k
ensemble on the OOD tiles. The drop in Spearman/r2 vs the full model = that band-group's OOD
importance. Reported overall AND for South Africa (the break) separately.

8 bands: 0-2 RGB, 3 NIR, 4-5 SWIR, 6 thermal, 7 nightlights.
Run on PC GPU0: PPY scripts/46_ood_channel_ablation.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

OOD = "data/processed/tile_cache_ood"; DEV = "cuda:0"; BS = 256
RUN = "results/cnn_full"; NORM = "data/processed/tile_cache_full/norm_stats.npz"
GROUPS = {"none": [], "RGB": [0, 1, 2], "NIR": [3], "SWIR": [4, 5], "thermal": [6], "nightlights": [7]}


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


@torch.no_grad()
def ensemble_predict(cache, ablate):
    stats = np.load(NORM); fps = []
    for fold in splits.fold_ids():
        m_ = stats[f"{fold}_mean"].reshape(1, 8, 1, 1); s_ = stats[f"{fold}_std"].reshape(1, 8, 1, 1)
        net = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
        net.load_state_dict(torch.load(f"{RUN}/model_fold{fold}.pt", map_location=DEV)); net.eval()
        pr = []
        for i0 in range(0, len(cache), BS):
            x = (np.asarray(cache[i0:i0+BS], dtype="float32") - m_) / s_
            if ablate:
                x[:, ablate, :, :] = 0.0       # normalised mean = 0 -> removes the band's signal
            pr.append(net(torch.from_numpy(x).to(DEV)).cpu().numpy())
        fps.append(np.concatenate(pr))
    return np.mean(fps, axis=0)


def main():
    cache = np.load(f"{OOD}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{OOD}/cache_metadata.csv")
    y = meta.wealth_index_mean.values.astype("float32"); za = meta.country.values == "ZA"
    out = {}
    print("=== OOD channel ablation (Spearman/r2 with each band-group removed) ===")
    print(f"{'ablate':<13}{'overall Spr':>12}{'overall r2':>12}{'S.Africa Spr':>14}")
    base = None
    for name, ch in GROUPS.items():
        p = ensemble_predict(cache, ch)
        sp = float(spearmanr(y, p)[0]); r2v = float(r2(y, p)); sp_za = float(spearmanr(y[za], p[za])[0])
        out[name] = dict(spearman=round(sp, 3), r2=round(r2v, 3), spearman_ZA=round(sp_za, 3))
        if name == "none":
            base = out[name]
        d = "" if name == "none" else f"  (ΔSpr {sp-base['spearman']:+.3f})"
        print(f"{name:<13}{sp:>+12.3f}{r2v:>+12.3f}{sp_za:>+14.3f}{d}")
    # importance = drop when removed
    print("\nband-group OOD importance (overall Spearman drop when removed):")
    for name in GROUPS:
        if name == "none":
            continue
        print(f"  {name:<12} {base['spearman']-out[name]['spearman']:+.3f}")
    json.dump(out, open("results/ood_channel_ablation.json", "w"), indent=2)
    print("wrote results/ood_channel_ablation.json")


if __name__ == "__main__":
    main()
