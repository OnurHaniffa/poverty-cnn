"""Adversarial validation — quantify HOW far OOD each country is, and whether that predicts
the accuracy drop. In the frozen 36k model's 512-d feature space, train a classifier to tell
TRAINING tiles from each OOD country's tiles (5-fold CV AUC): ~0.5 = indistinguishable (model
should transfer), ~1.0 = big distribution shift (expect breakage). Then correlate per-country
AUC with the OOD Spearman drop — if positive, the model breaks *in proportion to* measurable shift.

Run on PC GPU0 (after scripts/43): PPY scripts/44_ood_adversarial.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from poverty_cnn.models.poverty_resnet import PovertyResNet

DEV = "cuda:0"; BS = 256; FOLD = "A"; N_TRAIN = 3000; ID_SPEARMAN = 0.760
FULL_CACHE = "data/processed/tile_cache_full"; OOD_CACHE = "data/processed/tile_cache_ood"
NORM = "data/processed/tile_cache_full/norm_stats.npz"
NAMES = {"ZA": "S.Africa", "NM": "Namibia", "GA": "Gabon", "SZ": "Eswatini", "MD": "Madagascar", "NI": "Niger"}


@torch.no_grad()
def feats(cache, idx, mean, std, model):
    out = []
    for i0 in range(0, len(idx), BS):
        rows = idx[i0:i0+BS]
        x = np.stack([np.asarray(cache[int(r)], dtype="float32") for r in rows])
        x = (x - mean) / std
        out.append(model(torch.from_numpy(x).to(DEV)).cpu().numpy())
    return np.concatenate(out)


def main():
    stats = np.load(NORM)
    mean = stats[f"{FOLD}_mean"].reshape(1, 8, 1, 1); std = stats[f"{FOLD}_std"].reshape(1, 8, 1, 1)
    model = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
    model.load_state_dict(torch.load(f"results/cnn_full/model_fold{FOLD}.pt", map_location=DEV))
    model.net.fc = nn.Identity(); model.eval()   # -> 512-d feature extractor

    fmeta = pd.read_csv(f"{FULL_CACHE}/cache_metadata.csv"); fcache = np.load(f"{FULL_CACHE}/cache.npy", mmap_mode="r")
    rng = np.random.default_rng(0); tr_idx = rng.choice(len(fmeta), min(N_TRAIN, len(fmeta)), replace=False)
    Ftr = feats(fcache, tr_idx, mean, std, model)

    ometa = pd.read_csv(f"{OOD_CACHE}/cache_metadata.csv"); ocache = np.load(f"{OOD_CACHE}/cache.npy", mmap_mode="r")
    Food = feats(ocache, np.arange(len(ometa)), mean, std, model)

    res = {}
    print("=== adversarial validation (5-fold CV AUC: 0.5=same dist, 1.0=far OOD) ===")
    for c in sorted(ometa.country.unique()):
        ci = np.where(ometa.country.values == c)[0]
        n = min(len(ci), len(tr_idx))
        X = np.vstack([Ftr[:n], Food[ci][:n]]); ytrue = np.r_[np.zeros(n), np.ones(n)]
        auc = float(cross_val_score(LogisticRegression(max_iter=1000), X, ytrue, cv=5, scoring="roc_auc").mean())
        res[c] = round(auc, 3)
        print(f"  {NAMES.get(c,c):11} AUC {auc:.3f}  (n={n})")

    try:
        ood = json.load(open("results/ood_frozen_test.json"))
        drop = {c: ID_SPEARMAN - ood["36k_full"]["per_country"][c]["spearman"] for c in res}
        cs = sorted(res); a = [res[c] for c in cs]; d = [drop[c] for c in cs]
        r = float(np.corrcoef(a, d)[0, 1])
        print(f"\n  corr(adversarial AUC, in-dist Spearman drop) = {r:+.3f}")
        print("  positive => the model breaks IN PROPORTION TO measurable distribution shift (the expected, strong story)")
        res["_corr_auc_vs_drop"] = r
    except Exception:
        print("  (run scripts/43 first for the AUC-vs-drop correlation)")
    json.dump(res, open("results/ood_adversarial.json", "w"), indent=2)
    print("wrote results/ood_adversarial.json")


if __name__ == "__main__":
    main()
