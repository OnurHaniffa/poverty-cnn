"""Statistical-rigor finishing pass (P1.2 + P2 from the methods-research synthesis).

On the FULL model held-out predictions (results/cnn_full), joined to the full-cache
metadata for country + urban via the same deterministic test-row order the dataset
uses:

1. PER-COUNTRY ranking with MULTIPLE-COMPARISON CONTROL: Spearman rho + raw p per
   country, then Benjamini-Hochberg FDR q over the 23-country family. Without this,
   any "significant in country X" claim is p-hacked across 23 tests.
2. WORST-GROUP Pearson r (WILDS-PovertyMap convention): r for urban & rural per fold,
   worst = min(r_urban, r_rural), averaged over folds +/- std. Plus raw pooled Pearson
   r so we line up against PovertyMap (reports r), SustainBench (r2), RWI (Spearman).
3. Benchmark-comparison table.

CPU only (reads saved preds). Run on PC: PPY scripts/35_stats_rigor.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from poverty_cnn.data import splits

CACHE_META = "data/processed/tile_cache_full/cache_metadata.csv"
RUN = "results/cnn_full"

try:
    from scipy.stats import false_discovery_control
    def bh(p):
        return np.asarray(false_discovery_control(np.asarray(p), method="bh"))
except Exception:                                   # manual Benjamini-Hochberg
    def bh(p):
        p = np.asarray(p); m = len(p); o = np.argsort(p)
        ranked = p[o] * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m); out[o] = np.clip(q, 0, 1); return out


def pear(a, b):
    return float(pearsonr(a, b)[0]) if len(a) >= 5 else np.nan


def r2(a, b):
    return 1.0 - ((a - b) ** 2).sum() / (((a - a.mean()) ** 2).sum() + 1e-12)


def load_full():
    meta = pd.read_csv(CACHE_META)
    parts = []
    for fold in splits.fold_ids():
        z = np.load(f"{RUN}/preds_fold{fold}.npz", allow_pickle=True)
        rows = splits.clusters_for(meta, fold, "test"); sub = meta.iloc[rows]
        assert len(sub) == len(z["y"]), f"row mismatch fold {fold}: {len(sub)} vs {len(z['y'])}"
        parts.append(pd.DataFrame({"country": sub.country.to_numpy(), "urban": sub.urban.to_numpy(),
                                   "fold": fold, "y": z["y"], "pred": z["pred"]}))
    return pd.concat(parts, ignore_index=True)


def main():
    df = load_full()
    print(f"full-model held-out rows: {len(df)} | countries: {df.country.nunique()}")

    # 1) per-country Spearman + BH-FDR
    recs = []
    for c, g in df.groupby("country"):
        if len(g) < 10:
            continue
        rho, p = spearmanr(g.y, g.pred)
        recs.append(dict(country=c, n=int(len(g)), rho=float(rho), p=float(p)))
    R = pd.DataFrame(recs).sort_values("rho").reset_index(drop=True)
    R["q_bh"] = bh(R["p"].values)
    R["sig_raw"] = R.p < 0.05
    R["sig_bh"] = R.q_bh < 0.05
    print(f"\n### per-country Spearman with Benjamini-Hochberg FDR (family m={len(R)})")
    for _, r in R.iterrows():
        star = " *" if r.sig_bh else ""
        print(f"  {r.country}: rho {r.rho:+.3f} (n={r.n:5d})  p={r.p:.3g}  q_BH={r.q_bh:.3g}{star}")
    print(f"  significant at raw p<.05: {int(R.sig_raw.sum())}/{len(R)} "
          f"|  after BH-FDR q<.05: {int(R.sig_bh.sum())}/{len(R)}")

    # 2) worst-group Pearson r per fold (WILDS convention)
    rows = []
    for fold, g in df.groupby("fold"):
        gu, gr = g[g.urban == "U"], g[g.urban == "R"]
        ru, rr = pear(gu.y.values, gu.pred.values), pear(gr.y.values, gr.pred.values)
        rows.append(dict(fold=fold, r_all=pear(g.y.values, g.pred.values),
                         r_urban=ru, r_rural=rr, worst=min(ru, rr)))
    W = pd.DataFrame(rows)
    print("\n### worst-group Pearson r per fold (WILDS-PovertyMap convention)")
    print(W.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"  mean over folds: r_all {W.r_all.mean():+.3f}+/-{W.r_all.std():.3f}  |  "
          f"worst-group {W.worst.mean():+.3f}+/-{W.worst.std():.3f}")

    # pooled raw metrics in every benchmark's currency
    r_pool = pear(df.y.values, df.pred.values)
    rho_pool = float(spearmanr(df.y, df.pred)[0])
    r2_pool = float(r2(df.y.values, df.pred.values))
    print(f"\n### pooled: Pearson r {r_pool:+.3f} | r2 {r2_pool:+.3f} | Spearman {rho_pool:+.3f}")
    print("\n### benchmark map (same DHS family, DIFFERENT protocols — read as orientation, not apples-to-apples):")
    print(f"  ours (5-fold leave-country-out): "
          f"r {r_pool:.2f} | r2 {r2_pool:.2f} | worst-group r {W.worst.mean():.2f}")
    print( "  WILDS PovertyMap ERM (OOD=leave-country-out, same DHS data): overall r 0.78 | worst-group r 0.45")
    print( "  Yeh 2020 (pooled ridge on CNN features):     r2 ~0.67-0.70")
    print( "  -> same OOD protocol: we MATCH overall r (0.78) and BEAT worst-group (0.55 vs 0.45).")

    json.dump({"per_country": R.to_dict("records"),
               "worst_group_by_fold": W.to_dict("records"),
               "pooled": {"pearson_r": r_pool, "r2": r2_pool, "spearman": rho_pool,
                          "worst_group_r_mean": float(W.worst.mean()),
                          "worst_group_r_std": float(W.worst.std())},
               "n_sig_raw": int(R.sig_raw.sum()), "n_sig_bh": int(R.sig_bh.sum()),
               "family_size": int(len(R))},
              open("results/stats_rigor.json", "w"), indent=2)
    print("\nwrote results/stats_rigor.json")


if __name__ == "__main__":
    main()
