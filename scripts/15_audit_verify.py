"""Adversarial verification of the fairness audit BEFORE we draw figures.

Two things we must not get wrong:
1. Is the urban/rural r2/Spearman gap REAL unfairness, or a measurement artifact
   of rural villages being more homogeneous (less wealth spread -> mechanically
   lower r2/Spearman even at equal accuracy)? Check within-group wealth std + MAE.
2. Does NL-only have the SAME fairness profile as MS+NL? (Are the disparities
   driven by the nightlights signal itself?)

Usage (PC): PPY scripts/15_audit_verify.py
"""
from __future__ import annotations

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


def load(d, meta):
    cs, us, ys, ps = [], [], [], []
    for fold in splits.fold_ids():
        z = np.load(f"{d}/preds_fold{fold}.npz", allow_pickle=True)
        rows = splits.clusters_for(meta, fold, "test")
        sub = meta.iloc[rows]
        assert (sub["country"].to_numpy() == z["country"].astype(str)).all()
        cs.append(sub["country"].to_numpy()); us.append(sub["urban"].to_numpy())
        ys.append(z["y"]); ps.append(z["pred"])
    return pd.DataFrame({"country": np.concatenate(cs), "urban": np.concatenate(us),
                         "y": np.concatenate(ys), "pred": np.concatenate(ps)})


def grp(df, k, v):
    m = df[df[k] == v]; y, p = m["y"].to_numpy(), m["pred"].to_numpy()
    return dict(n=len(m), wealth_mean=y.mean(), wealth_std=y.std(),
               r2=r2(y, p), mae=mae(y, p), spear=spear(y, p))


def main():
    meta = pd.read_csv(META)
    full = load("results/cnn_stable", meta)   # MS+NL seed42
    nl = load("results/cnn_nl_stable", meta)   # NL-only seed42

    print("### CHECK 1 — is the urban/rural gap real, or a homogeneity artifact? (MS+NL)")
    u, r = grp(full, "urban", "U"), grp(full, "urban", "R")
    print(f"  urban: n={u['n']:5} wealth_std={u['wealth_std']:.3f}  r2={u['r2']:+.3f}  MAE={u['mae']:.3f}  spear={u['spear']:+.3f}")
    print(f"  rural: n={r['n']:5} wealth_std={r['wealth_std']:.3f}  r2={r['r2']:+.3f}  MAE={r['mae']:.3f}  spear={r['spear']:+.3f}")
    print(f"  --> urban wealth_std/rural wealth_std = {u['wealth_std']/r['wealth_std']:.2f}")
    print( "  --> if urban spread >> rural spread, the r2/Spearman gap is PARTLY compression, not pure unfairness.")
    print(f"  --> MAE gap (urban-rural) = {u['mae']-r['mae']:+.3f}  (near 0 = equal ABSOLUTE accuracy)")

    print("\n### CHECK 2 — NL-only vs MS+NL: same fairness profile? (seed42, apples-to-apples)")
    for lab, df in [("MS+NL", full), ("NL-only", nl)]:
        u2, r2g = grp(df, "urban", "U"), grp(df, "urban", "R")
        print(f"  {lab:7}: urban spear {u2['spear']:+.3f} | rural spear {r2g['spear']:+.3f} | "
              f"urban-rural spear gap {u2['spear']-r2g['spear']:+.3f}")

    # per-country r2 correlation between the two channel configs
    cc = sorted(full["country"].unique())
    fr = [grp(full, "country", c)["r2"] for c in cc]
    nr = [grp(nl, "country", c)["r2"] for c in cc]
    print(f"\n  per-country r2 correlation MS+NL vs NL-only: {np.corrcoef(fr, nr)[0,1]:+.3f}")
    print( "  --> high correlation = the SAME countries are well/poorly served either way")
    print( "      (i.e. the disparities are driven by the nightlights signal, not daytime imagery)")
    # who flips
    print("\n  biggest per-country differences (NL-only r2 minus MS+NL r2):")
    diffs = sorted(zip(cc, np.array(nr) - np.array(fr)), key=lambda x: x[1])
    for c, d in diffs[:3] + diffs[-3:]:
        print(f"    {c}: {d:+.3f}  (MS+NL {dict(zip(cc,fr))[c]:+.3f} -> NL-only {dict(zip(cc,nr))[c]:+.3f})")


if __name__ == "__main__":
    main()
