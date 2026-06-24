"""Bootstrap 95% confidence intervals for a trained run's held-out metrics.

Resamples the saved per-cluster held-out predictions with replacement to put
error bars on pooled r2 / MAE / Spearman and on mean-of-folds r2. This is what
turns a single point estimate into a defensible, publishable number.

Usage (on the PC, preds live there; no GPU needed):
  PPY scripts/11_bootstrap_ci.py results/cnn_stable
  PPY scripts/11_bootstrap_ci.py results/cnn_stable --n 2000
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

try:
    from scipy.stats import spearmanr
    def spear(a, b):
        return float(spearmanr(a, b)[0])
except Exception:
    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


def r2(a, b):
    return 1.0 - ((a - b) ** 2).sum() / (((a - a.mean()) ** 2).sum() + 1e-12)


def mae(a, b):
    return float(np.abs(a - b).mean())


def ci(vals):
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))


def load_folds(d):
    """Return list of (y, pred, country) per fold + concatenated pooled arrays."""
    folds = []
    for f in sorted(glob.glob(d + "/preds_fold*.npz")):
        z = np.load(f, allow_pickle=True)
        folds.append((z["y"], z["pred"], z["country"].astype(str)))
    y = np.concatenate([f[0] for f in folds])
    p = np.concatenate([f[1] for f in folds])
    c = np.concatenate([f[2] for f in folds])
    return folds, y, p, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    folds, y, p, c = load_folds(args.run)
    n = len(y)
    print(f"=== bootstrap 95% CI for {args.run}  (n={n}, {args.n} resamples) ===")
    print(f"point estimates: pooled r2={r2(y,p):.3f}  MAE={mae(y,p):.3f}  spearman={spear(y,p):.3f}")

    # ---- i.i.d. (naive) pooled bootstrap: resample individual clusters ----
    pr2, pmae, psp, pmean = [], [], [], []
    for _ in range(args.n):
        idx = rng.integers(0, n, n)              # resample pooled with replacement
        yb, pb = y[idx], p[idx]
        pr2.append(r2(yb, pb)); pmae.append(mae(yb, pb)); psp.append(spear(yb, pb))
        # mean-of-folds: resample WITHIN each fold, average per-fold r2
        fr2 = []
        for fy, fp, _ in folds:
            j = rng.integers(0, len(fy), len(fy))
            fr2.append(r2(fy[j], fp[j]))
        pmean.append(np.mean(fr2))

    # ---- CLUSTER (whole-country) bootstrap: the honest CI given between>>within ----
    # Our wealth signal is mostly BETWEEN countries, so clusters are not independent.
    # Resample the LIST of countries with replacement, gather their rows -> wider, correct CIs.
    countries = np.unique(c)
    crows = {k: np.where(c == k)[0] for k in countries}
    cr2, cmae, csp = [], [], []
    for _ in range(args.n):
        pick = rng.choice(countries, size=len(countries), replace=True)
        idx = np.concatenate([crows[k] for k in pick])
        yb, pb = y[idx], p[idx]
        cr2.append(r2(yb, pb)); cmae.append(mae(yb, pb)); csp.append(spear(yb, pb))

    print(f"\n-- i.i.d. cluster bootstrap (naive, too narrow under spatial structure) --")
    for name, vals in [("pooled r2", pr2), ("pooled MAE", pmae),
                       ("pooled Spearman", psp), ("mean-of-folds r2", pmean)]:
        lo, md, hi = ci(vals)
        print(f"  {name:18s}: {md:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    print(f"\n-- whole-country (cluster) bootstrap (HONEST headline CI, m={len(countries)} countries) --")
    for name, vals in [("pooled r2", cr2), ("pooled MAE", cmae), ("pooled Spearman", csp)]:
        lo, md, hi = ci(vals)
        print(f"  {name:18s}: {md:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
