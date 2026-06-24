"""Frozen-model OOD test — the capstone. No retraining, no leakage.

For each model (36k full, 13k pilot), ensemble its 5 cross-country fold-models (none of which
saw any OOD country), normalising the OOD tiles with each fold's OWN training norm stats, and
score per-country r2/MAE/Spearman/Pearson vs the frozen-PCA wealth. Then compare 13k vs 36k:
does 3x training data improve out-of-distribution generalisation?

Labels are on the 36k multiround PCA axis. For the 36k model this is exact; for the 13k pilot
(single-round axis) Spearman (rank, scale-invariant) is the fair cross-model metric.

Run on PC GPU0: PPY scripts/43_ood_frozen_test.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

OOD = "data/processed/tile_cache_ood"; DEV = "cuda:0"; BS = 256
ID_SPEARMAN_FULL = 0.760   # the full model's in-distribution pooled Spearman (held-out 23 countries)
NAMES = {"ZA": "S.Africa", "NM": "Namibia", "GA": "Gabon", "SZ": "Eswatini", "MD": "Madagascar", "NI": "Niger"}
MODELS = {
    "36k_full":  dict(run="results/cnn_full",   norm="data/processed/tile_cache_full/norm_stats.npz", axis="multiround axis (exact)"),
    "13k_pilot": dict(run="results/cnn_stable", norm="data/processed/tile_cache/norm_stats.npz",      axis="single-round axis (use Spearman to compare)"),
}
TEAL = "#178a7a"; AMBER = "#e0922f"; NAVY = "#1f3a5f"; GREY = "#9aa4ae"


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


@torch.no_grad()
def ensemble_predict(cache, run, normf):
    stats = np.load(normf)
    fold_preds = []
    for fold in splits.fold_ids():
        mean = stats[f"{fold}_mean"].reshape(1, 8, 1, 1); std = stats[f"{fold}_std"].reshape(1, 8, 1, 1)
        m = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
        m.load_state_dict(torch.load(f"{run}/model_fold{fold}.pt", map_location=DEV)); m.eval()
        preds = []
        for i0 in range(0, len(cache), BS):
            x = (np.asarray(cache[i0:i0+BS], dtype="float32") - mean) / std
            preds.append(m(torch.from_numpy(x).to(DEV)).cpu().numpy())
        fold_preds.append(np.concatenate(preds))
    return np.mean(fold_preds, axis=0)


def metrics(y, p):
    return dict(r2=round(float(r2(y, p)), 3), mae=round(float(np.abs(y - p).mean()), 3),
                spearman=round(float(spearmanr(y, p)[0]), 3),
                pearson=round(float(pearsonr(y, p)[0]), 3), n=int(len(y)))


def main():
    cache = np.load(f"{OOD}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{OOD}/cache_metadata.csv")
    y = meta.wealth_index_mean.values.astype("float32"); cc = meta.country.values
    out = {}
    for name, cfg in MODELS.items():
        pred = ensemble_predict(cache, cfg["run"], cfg["norm"])
        overall = metrics(y, pred)
        per = {c: metrics(y[cc == c], pred[cc == c]) for c in sorted(np.unique(cc))}
        out[name] = dict(axis=cfg["axis"], overall=overall, per_country=per)
        print(f"\n=== {name}  [{cfg['axis']}] ===")
        print(f"  OVERALL: r2 {overall['r2']:+.3f} | MAE {overall['mae']:.3f} | "
              f"Spearman {overall['spearman']:+.3f} | Pearson {overall['pearson']:+.3f}  (vs in-dist Spearman {ID_SPEARMAN_FULL})")
        for c, m in sorted(per.items(), key=lambda kv: -kv[1]['spearman']):
            print(f"    {NAMES.get(c,c):11} Spearman {m['spearman']:+.3f} | r2 {m['r2']:+.3f} | MAE {m['mae']:.3f} | n={m['n']}")

    print("\n=== does 3x training data improve OOD generalisation? (per-country Spearman) ===")
    for c in sorted(np.unique(cc)):
        a = out["13k_pilot"]["per_country"][c]["spearman"]; b = out["36k_full"]["per_country"][c]["spearman"]
        print(f"  {NAMES.get(c,c):11} 13k {a:+.3f} -> 36k {b:+.3f}   (Δ {b-a:+.3f})")
    json.dump(out, open("results/ood_frozen_test.json", "w"), indent=2)

    # ---- figure: per-country Spearman, both models, vs in-distribution ----
    countries = sorted(np.unique(cc), key=lambda c: -out["36k_full"]["per_country"][c]["spearman"])
    x = np.arange(len(countries))
    s13 = [out["13k_pilot"]["per_country"][c]["spearman"] for c in countries]
    s36 = [out["36k_full"]["per_country"][c]["spearman"] for c in countries]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(ID_SPEARMAN_FULL, ls="--", color=GREY, lw=1.5, label=f"in-distribution (0.76)")
    ax.bar(x - 0.2, s13, 0.4, color=AMBER, label="13k model")
    ax.bar(x + 0.2, s36, 0.4, color=TEAL, label="36k model")
    ax.set_xticks(x); ax.set_xticklabels([NAMES.get(c, c) for c in countries], rotation=20)
    ax.set_ylabel("Spearman (ranking)"); ax.set_ylim(0, 1)
    ax.set_title("OOD generalisation: where does it break?  (frozen models, unseen countries)",
                 color=NAVY, fontweight="bold")
    ax.legend(frameon=False); ax.grid(True, axis="y", color="#eee")
    fig.tight_layout()
    Path("results/figures/teaching").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/teaching/ood_where_it_breaks.png", dpi=170, bbox_inches="tight", facecolor="white")
    print("\nwrote results/ood_frozen_test.json + ood_where_it_breaks.png")


if __name__ == "__main__":
    main()
