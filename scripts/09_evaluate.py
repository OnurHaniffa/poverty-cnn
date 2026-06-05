"""Evaluate trained runs: per-fold + per-country r2 / MAE / Spearman, and
aggregate across seeds (mean +/- std) to expose the per-country noise floor.

r2 alone misleads for the fairness audit (it conflates model error with a
country's wealth spread), so we always report MAE (absolute error) and Spearman
(ranking quality) alongside it.

Usage (on the PC, tiles+preds live there):
  PPY scripts/09_evaluate.py results/cnn_stable
  PPY scripts/09_evaluate.py results/cnn_stable results/cnn_stable_s1 results/cnn_stable_s2
"""
from __future__ import annotations

import glob
import sys

import numpy as np

try:
    from scipy.stats import spearmanr
    def spear(a, b):
        return float(spearmanr(a, b)[0])
except Exception:  # numpy fallback (ties handled approximately)
    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


def r2(a, b):
    return 1.0 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def mae(a, b):
    return float(np.abs(a - b).mean())


def load_run(d):
    ys, ps, cc = [], [], []
    for f in sorted(glob.glob(d + "/preds_fold*.npz")):
        z = np.load(f, allow_pickle=True)
        ys.append(z["y"]); ps.append(z["pred"]); cc.append(z["country"])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(cc).astype(str)


def by_country(y, p, c):
    out = {}
    for k in sorted(set(c)):
        m = c == k
        if m.sum() > 5:
            out[k] = (r2(y[m], p[m]), mae(y[m], p[m]), spear(y[m], p[m]), int(m.sum()))
    return out


def main():
    runs = sys.argv[1:] or ["results/cnn_stable"]
    per_run = {}
    print("=== per-run pooled (held-out, all countries) ===")
    for d in runs:
        y, p, c = load_run(d)
        per_run[d] = by_country(y, p, c)
        print("  %-26s pooled r2=%.3f  MAE=%.3f  spearman=%.3f  n=%d"
              % (d, r2(y, p), mae(y, p), spear(y, p), len(y)))

    countries = sorted(set.intersection(*[set(m) for m in per_run.values()]))
    print("\n=== per-country across %d run(s), sorted by mean r2 ===" % len(runs))
    rows = []
    for k in countries:
        r2s = [per_run[d][k][0] for d in runs]
        maes = [per_run[d][k][1] for d in runs]
        sps = [per_run[d][k][2] for d in runs]
        rows.append((k, np.mean(r2s), np.std(r2s), np.mean(maes), np.mean(sps),
                     per_run[runs[0]][k][3]))
    for k, rm, rs, mm, sm, n in sorted(rows, key=lambda x: x[1]):
        print("  %s  r2=%+.3f +/-%.3f   MAE=%.3f   spear=%+.3f   n=%4d"
              % (k, rm, rs, mm, sm, n))
    if len(runs) > 1:
        spread = np.mean([r[2] for r in rows])
        print("\nmean per-country r2 std across seeds (the NOISE FLOOR): %.3f" % spread)


if __name__ == "__main__":
    main()
