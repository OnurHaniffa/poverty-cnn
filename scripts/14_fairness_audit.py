"""Fairness audit (contribution #2): per-country + urban/rural, multi-metric,
across-seed noise floor.

Yeh reported one pooled r2. We audit WHO the model serves well vs poorly, across
all 23 countries and the urban/rural split, using three metrics that disagree
on purpose:
  - r2       : variance explained (but conflates error with a group's wealth spread)
  - MAE      : absolute error in wealth-index units (honest "how far off")
  - Spearman : ranking quality (what aid-targeting actually needs)

Predictions saved (y, pred, country) in test-fold order; we rejoin urban/rural by
replaying the deterministic test split and asserting the country labels line up.

Usage (PC): PPY scripts/14_fairness_audit.py
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from poverty_cnn.data import splits

try:
    from scipy.stats import spearmanr
    def spear(a, b): return float(spearmanr(a, b)[0])
except Exception:
    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

def r2(a, b): return 1.0 - ((a - b) ** 2).sum() / (((a - a.mean()) ** 2).sum() + 1e-12)
def mae(a, b): return float(np.abs(a - b).mean())

META = "data/processed/tile_cache/cache_metadata.csv"
SEEDS = {"s42": "results/cnn_stable", "s1": "results/cnn_stable_s1", "s2": "results/cnn_stable_s2"}


def load_seed(d, meta):
    """Return dataframe (country, urban, y, pred) for one seed, rejoined to urban/rural."""
    cs, us, ys, ps = [], [], [], []
    for fold in splits.fold_ids():
        z = np.load(f"{d}/preds_fold{fold}.npz", allow_pickle=True)
        rows = splits.clusters_for(meta, fold, "test")
        sub = meta.iloc[rows]
        assert len(rows) == len(z["y"]), f"{d} {fold}: length mismatch"
        assert (sub["country"].to_numpy() == z["country"].astype(str)).all(), \
            f"{d} {fold}: country alignment broken"
        cs.append(sub["country"].to_numpy()); us.append(sub["urban"].to_numpy())
        ys.append(z["y"]); ps.append(z["pred"])
    return pd.DataFrame({"country": np.concatenate(cs), "urban": np.concatenate(us),
                         "y": np.concatenate(ys), "pred": np.concatenate(ps)})


def metrics_by(df, key, val):
    m = df[df[key] == val]
    return r2(m["y"].to_numpy(), m["pred"].to_numpy()), mae(m["y"].to_numpy(), m["pred"].to_numpy()), \
        spear(m["y"].to_numpy(), m["pred"].to_numpy()), len(m), m["y"].std()


def agg(per_seed_vals):
    a = np.array(per_seed_vals)
    return a.mean(), a.std()


def classify(r2m, maem, spm, ystd):
    if r2m >= 0.45: return "well-served"
    if spm >= 0.70: return "miscalibrated (ranks OK, scale off)"
    if ystd < 0.65 and maem < 0.40: return "low-variance (little to predict)"
    return "GENUINE MISS"


def main():
    meta = pd.read_csv(META)
    seed_dfs = {s: load_seed(d, meta) for s, d in SEEDS.items()}
    print(f"loaded {len(seed_dfs)} seeds, alignment OK\n")

    # --- urban / rural ---
    print("=== URBAN vs RURAL (across-seed mean +/- std) ===")
    for grp, lab in [("U", "urban"), ("R", "rural")]:
        r2s = [metrics_by(df, "urban", grp)[0] for df in seed_dfs.values()]
        maes = [metrics_by(df, "urban", grp)[1] for df in seed_dfs.values()]
        sps = [metrics_by(df, "urban", grp)[2] for df in seed_dfs.values()]
        n = metrics_by(list(seed_dfs.values())[0], "urban", grp)[3]
        print(f"  {lab:5} (n={n:5}): r2 {agg(r2s)[0]:+.3f}+/-{agg(r2s)[1]:.3f}   "
              f"MAE {agg(maes)[0]:.3f}   spearman {agg(sps)[0]:+.3f}")

    # --- per country ---
    print("\n=== PER-COUNTRY (across-seed mean, sorted by r2) ===")
    countries = sorted(seed_dfs["s42"]["country"].unique())
    rows = []
    for c in countries:
        r2s = [metrics_by(df, "country", c)[0] for df in seed_dfs.values()]
        maes = [metrics_by(df, "country", c)[1] for df in seed_dfs.values()]
        sps = [metrics_by(df, "country", c)[2] for df in seed_dfs.values()]
        n, ystd = metrics_by(seed_dfs["s42"], "country", c)[3:5]
        rm, rs = agg(r2s); mm, _ = agg(maes); sm, _ = agg(sps)
        rows.append((c, rm, rs, mm, sm, n, ystd, classify(rm, mm, sm, ystd)))
    for c, rm, rs, mm, sm, n, ystd, cls in sorted(rows, key=lambda x: x[1]):
        print(f"  {c}  r2 {rm:+.3f}+/-{rs:.3f}  MAE {mm:.3f}  spear {sm:+.3f}  n={n:4}  [{cls}]")

    out = pd.DataFrame(rows, columns=["country", "r2_mean", "r2_std", "mae", "spearman",
                                      "n", "wealth_std", "class"])
    out.to_csv("results/fairness_audit_percountry.csv", index=False)
    print("\nsaved -> results/fairness_audit_percountry.csv")
    print("\nclass counts:", out["class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
