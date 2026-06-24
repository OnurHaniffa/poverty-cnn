"""Residual spatial-autocorrelation (Moran's I) on the full model's held-out residuals.

For each test country, build a k-NN graph on the cluster GPS and compute Moran's I of the
held-out residuals (y - pred), with a permutation p-value. Interpretation for OUR design:
because folds are leave-COUNTRY-out (train and test are *different* countries), any residual
SAC is UNMODELLED LOCAL STRUCTURE, not train-test leakage — leakage is impossible by
construction. This exhibit confirms the residuals aren't carrying a leakage signature and
reports how much spatial signal the model leaves on the table.

Run on PC: PPY scripts/48_residual_sac.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from poverty_cnn.data import splits

RUN = "results/cnn_full"; META = "data/processed/tile_cache_full/cache_metadata.csv"; K = 8


def morans_i(resid, coords, k=K, nperm=499, seed=0):
    n = len(resid)
    if n < k + 5:
        return None
    z = resid - resid.mean()
    nbr = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]    # k nearest, drop self
    denom = np.sum(z ** 2)
    # row-standardized weights (w_ij = 1/k) => N/S0 = 1, so I = Σ_ij w_ij z_i z_j / Σ z_i²
    I = (np.sum(z[:, None] * z[nbr]) / k) / denom
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(nperm):
        zp = rng.permutation(z)
        Ip = (np.sum(zp[:, None] * zp[nbr]) / k) / np.sum(zp ** 2)
        if abs(Ip) >= abs(I):
            cnt += 1
    return float(I), float((cnt + 1) / (nperm + 1)), int(n)


def main():
    meta = pd.read_csv(META)
    parts = []
    for fold in splits.fold_ids():
        z = np.load(f"{RUN}/preds_fold{fold}.npz", allow_pickle=True)
        rows = splits.clusters_for(meta, fold, "test"); sub = meta.iloc[rows]
        parts.append(pd.DataFrame({"country": sub.country.to_numpy(), "lat": sub.lat.to_numpy(),
                                   "lon": sub.lon.to_numpy(), "resid": z["y"] - z["pred"]}))
    df = pd.concat(parts, ignore_index=True).dropna(subset=["lat", "lon"])
    print(f"held-out residuals with GPS: {len(df)}")
    res = {}
    print("\n=== per-country residual Moran's I (k=8, 499 perms) ===")
    for c, g in df.groupby("country"):
        r = morans_i(g.resid.values, g[["lat", "lon"]].values)
        if r is None:
            continue
        I, p, n = r
        res[c] = dict(morans_I=round(I, 3), p=round(p, 3), n=n)
        sig = "*" if p < 0.05 else " "
        print(f"  {c}: I {I:+.3f}  p={p:.3f} {sig}  (n={n})")
    Is = [v["morans_I"] for v in res.values()]
    nsig = sum(v["p"] < 0.05 for v in res.values())
    print(f"\n  mean Moran's I {np.mean(Is):+.3f} | significant (p<.05): {nsig}/{len(res)}")
    print("  NOTE: folds are country-disjoint, so this is unmodelled LOCAL structure, NOT leakage")
    print("  (leakage would require train clusters near test clusters — impossible across countries).")
    json.dump(res, open("results/residual_sac.json", "w"), indent=2)
    print("wrote results/residual_sac.json")


if __name__ == "__main__":
    main()
