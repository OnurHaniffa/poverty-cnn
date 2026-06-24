"""Field-standard uncertainty-quantification metrics (replaces corr(unc,error)).

For each of our three UQ methods — deep ensemble (std over 5 seeds), MC-dropout
(mc_std), heteroscedastic (sigma) — compute the metrics that the deep-regression UQ
literature actually uses (AUSE/AURG, NLL, calibration/ENCE/PICP, variance-scaling
factor), pooled over the 5 cross-country folds. CPU-only (reads saved .npz preds —
does NOT touch the GPU).

Why these, in one line each:
- AUSE  = Area Under the Sparsification Error: drop the most-uncertain points; does
          RMSE fall as fast as if we dropped the truly-worst points (the oracle)? Gap
          to oracle, LOWER is better.
- AURG  = Area Under the Random Gain: how much better than dropping points at random.
          HIGHER is better; ~0 means the uncertainty is no better than a coin flip.
- NLL   = Gaussian negative log-likelihood (proper scoring rule). LOWER is better.
- ENCE  = Expected Normalised Calibration Error: are the error bars the right size?
- PICP  = fraction of truths inside the nominal 90% interval (want ~0.90); MPIW = its width.
- s     = variance-scaling factor sqrt(mean(((y-mean)/sigma)^2)); s>1 underconfident
          (bars too small), s<1 overconfident, s~1 calibrated.
- equity: nominal-90% coverage for the POOREST-20% vs RICHEST-20% — group-conditional
          calibration. Our thesis in field-standard form: are the poor's error bars honest?

Run on PC (CPU): PPY scripts/34_uncertainty_proper_metrics.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FOLDS = ["A", "B", "C", "D", "E"]
ENSEMBLE_RUNS = ["results/cnn_stable", "results/cnn_stable_s1", "results/cnn_stable_s2",
                 "results/cnn_stable_s3", "results/cnn_stable_s4"]
NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; PLUM="#7a4ea8"; GREY="#9aa4ae"
OUT=Path("results/figures/teaching"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"Avenir Next","axes.spines.top":False,"axes.spines.right":False})


def load_method():
    """Return dict method -> (y, mean, sigma) pooled over folds."""
    out = {}
    # ---- deep ensemble: mean & std across the 5 seed runs ----
    ey, em, es = [], [], []
    for f in FOLDS:
        preds = np.stack([np.load(f"{r}/preds_fold{f}.npz")["pred"] for r in ENSEMBLE_RUNS])  # (5, N)
        y = np.load(f"{ENSEMBLE_RUNS[0]}/preds_fold{f}.npz")["y"]
        ey.append(y); em.append(preds.mean(0)); es.append(preds.std(0, ddof=1))
    out["Deep ensemble"] = (np.concatenate(ey), np.concatenate(em), np.concatenate(es))
    # ---- MC-dropout ----
    my, mm, ms = [], [], []
    for f in FOLDS:
        z = np.load(f"results/cnn_stable/mc_preds_fold{f}.npz")
        my.append(z["y"]); mm.append(z["mc_mean"]); ms.append(z["mc_std"])
    out["MC-dropout"] = (np.concatenate(my), np.concatenate(mm), np.concatenate(ms))
    # ---- heteroscedastic ----
    hy, hm, hs = [], [], []
    for f in FOLDS:
        z = np.load(f"results/cnn_hetero/preds_fold{f}.npz")
        hy.append(z["y"]); hm.append(z["mean"]); hs.append(z["sigma"])
    out["Heteroscedastic"] = (np.concatenate(hy), np.concatenate(hm), np.concatenate(hs))
    return out


def sparsification(err, order, fracs):
    """RMSE of the points REMAINING after removing the top-f by `order` (worst first)."""
    e = err[order]; N = len(e)
    return np.array([np.sqrt((e[int(f*N):]**2).mean()) for f in fracs])


def ause_aurg(y, mean, sigma):
    err = np.abs(y - mean)
    fracs = np.linspace(0.0, 0.95, 96)
    by_unc = sparsification(err, np.argsort(-sigma), fracs)     # our uncertainty ranks removal
    by_orc = sparsification(err, np.argsort(-err),   fracs)     # oracle: remove true-worst
    rmse0 = by_unc[0]                                            # full-set RMSE (normaliser)
    random_curve = np.full_like(fracs, rmse0)                   # random removal ~ flat at full RMSE
    spars_err = (by_unc - by_orc) / rmse0                       # normalised gap to oracle
    rand_gain = (random_curve - by_unc) / rmse0                 # normalised gain over random
    ause = float(np.trapz(spars_err, fracs))
    aurg = float(np.trapz(rand_gain, fracs))
    return ause, aurg, fracs, by_unc/rmse0, by_orc/rmse0


def calibration(y, mean, sigma):
    z = (y - mean) / np.clip(sigma, 1e-9, None)
    nll = float(np.mean(0.5*np.log(2*np.pi*sigma**2) + 0.5*z**2))
    s = float(np.sqrt(np.mean(z**2)))                           # variance-scaling factor
    levels = np.linspace(0.05, 0.95, 19)
    emp = np.array([np.mean(np.abs(z) <= norm.ppf(0.5+p/2)) for p in levels])  # central coverage
    ence = float(np.mean(np.abs(emp - levels)))
    # 90% prediction interval
    z90 = norm.ppf(0.95)
    inside90 = np.abs(z) <= z90
    picp90 = float(inside90.mean()); mpiw90 = float((2*z90*sigma).mean())
    # group-conditional coverage @90% : poorest-20% vs richest-20%
    qlo, qhi = np.quantile(y, 0.20), np.quantile(y, 0.80)
    cov_poor = float(inside90[y <= qlo].mean()); cov_rich = float(inside90[y >= qhi].mean())
    return dict(nll=nll, scale_factor=s, ence=ence, picp90=picp90, mpiw90=mpiw90,
                cov_poorest20=cov_poor, cov_richest20=cov_rich,
                levels=levels.tolist(), emp_coverage=emp.tolist())


def main():
    methods = load_method()
    colors = {"Deep ensemble": TEAL, "MC-dropout": AMBER, "Heteroscedastic": PLUM}
    results = {}
    sp_curves = {}

    print("="*78)
    print(f"{'method':<18}{'AUSE↓':>9}{'AURG↑':>9}{'NLL↓':>9}{'ENCE↓':>9}{'PICP90':>9}{'scale s':>9}")
    print("-"*78)
    for name, (y, mean, sigma) in methods.items():
        ause, aurg, fracs, c_unc, c_orc = ause_aurg(y, mean, sigma)
        cal = calibration(y, mean, sigma)
        old_corr = float(np.corrcoef(sigma, np.abs(y-mean))[0, 1])  # the OLD metric, for contrast
        results[name] = dict(ause=ause, aurg=aurg, **cal, old_corr_unc_err=old_corr)
        sp_curves[name] = (fracs, c_unc, c_orc)
        print(f"{name:<18}{ause:>9.4f}{aurg:>9.4f}{cal['nll']:>9.3f}{cal['ence']:>9.4f}"
              f"{cal['picp90']:>9.3f}{cal['scale_factor']:>9.3f}")
    print("-"*78)
    print(f"{'poorest-20% vs richest-20% coverage @ nominal 90%:':<50}")
    for name in methods:
        r = results[name]
        print(f"  {name:<18} poorest {r['cov_poorest20']:.3f} | richest {r['cov_richest20']:.3f} "
              f"| gap {r['cov_richest20']-r['cov_poorest20']:+.3f}")
    print(f"\n(for contrast, the OLD metric corr(unc,|err|): "
          + ", ".join(f"{n} {results[n]['old_corr_unc_err']:+.3f}" for n in methods) + ")")

    json.dump(results, open("results/uncertainty_proper.json", "w"), indent=2)

    # ---------- FIG 1: sparsification curves ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, (fr, c_unc, c_orc) in sp_curves.items():
        ax.plot(fr, c_unc, "-", color=colors[name], lw=2.3, label=f"{name} (AUSE={results[name]['ause']:.3f})")
    # one oracle (same target for all) + random
    any_orc = next(iter(sp_curves.values()))[2]
    ax.plot(fr, any_orc, "--", color=NAVY, lw=1.8, label="oracle (perfect ranking)")
    ax.axhline(1.0, ls=":", color=GREY, lw=1.5, label="random removal")
    ax.set_xlabel("fraction of most-uncertain villages removed")
    ax.set_ylabel("RMSE of remainder  (÷ full-set RMSE)")
    ax.set_title("Sparsification: does uncertainty find the bad predictions?", color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(True, color="#eee")
    fig.tight_layout(); fig.savefig(OUT/"uq_sparsification.png", dpi=180, bbox_inches="tight", facecolor="white")

    # ---------- FIG 2: calibration reliability diagram ----------
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.5, label="perfectly calibrated")
    for name in methods:
        r = results[name]
        ax.plot(r["levels"], r["emp_coverage"], "-o", color=colors[name], lw=2, ms=4,
                label=f"{name} (ENCE={r['ence']:.3f}, s={r['scale_factor']:.2f})")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration reliability diagram", color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.set_aspect("equal"); ax.grid(True, color="#eee")
    fig.tight_layout(); fig.savefig(OUT/"uq_calibration.png", dpi=180, bbox_inches="tight", facecolor="white")

    print("\nwrote results/uncertainty_proper.json + uq_sparsification.png + uq_calibration.png")


if __name__ == "__main__":
    main()
