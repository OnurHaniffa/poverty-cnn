"""Constructive flip of the UQ negative: does a TRIVIAL domain flag — 'is this village dark
at night?' — flag the model's unreliable predictions BETTER than the learned uncertainty methods?

Our headline negative: ensembles / MC-dropout / heteroscedastic UQ barely beat random at ranking
errors (AURG ≈ 0). But night-lights are dark exactly where the model fails. So we test
night-light darkness (reliability ∝ −NL) as an error-ranker on the same pilot predictions, and
compare its AURG / corr-with-error to the three UQ methods. If darkness wins, the practical message
becomes: 'forget calibrated UQ — where the map is dark, don't trust it.'

Run on PC: PPY scripts/53_reliability_flag.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from poverty_cnn.data import splits

CACHE = "data/processed/tile_cache"; NL = 7   # pilot cache, nightlights band
FOLDS = ["A", "B", "C", "D", "E"]
ENS = ["results/cnn_stable", "results/cnn_stable_s1", "results/cnn_stable_s2", "results/cnn_stable_s3", "results/cnn_stable_s4"]
NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; PLUM="#7a4ea8"; RED="#c0504d"; GREY="#9aa4ae"
OUT = Path("results/figures/uncertainty"); OUT.mkdir(parents=True, exist_ok=True)


def aurg(err, unc, fracs):
    """Area under random-gain: how much better than random at shedding error. Higher=better."""
    e = err[np.argsort(-unc)]; N = len(e)
    curve = np.array([np.sqrt((e[int(f*N):]**2).mean()) for f in fracs])
    return float(np.trapz((curve[0] - curve)/curve[0], fracs)), curve/curve[0]


def main():
    cache = np.load(f"{CACHE}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{CACHE}/cache_metadata.csv")
    # per-tile nightlight mean + ensemble error + hetero/MC sigmas, aligned over folds
    nl, y, em, es, hs, ms = [], [], [], [], [], []
    for f in FOLDS:
        rows = splits.clusters_for(meta, f, "test")
        nl.append(np.array([float(np.asarray(cache[int(r), NL]).mean()) for r in rows]))
        preds = np.stack([np.load(f"{r}/preds_fold{f}.npz")["pred"] for r in ENS])
        yy = np.load(f"{ENS[0]}/preds_fold{f}.npz")["y"]
        y.append(yy); em.append(preds.mean(0)); es.append(preds.std(0, ddof=1))
        hs.append(np.load(f"results/cnn_hetero/preds_fold{f}.npz")["sigma"])
        ms.append(np.load(f"results/cnn_stable/mc_preds_fold{f}.npz")["mc_std"])
    nl = np.concatenate(nl); y = np.concatenate(y); em = np.concatenate(em)
    es = np.concatenate(es); hs = np.concatenate(hs); ms = np.concatenate(ms)
    err = np.abs(y - em)
    fracs = np.linspace(0.0, 0.95, 96)

    methods = {
        "Night-light darkness (−NL)": -nl,
        "Deep ensemble σ": es,
        "MC-dropout σ": ms,
        "Heteroscedastic σ": hs,
    }
    res = {}
    print("=== reliability flag: AURG (↑ better) + corr with |error| ===")
    curves = {}
    for name, unc in methods.items():
        a, c = aurg(err, unc, fracs)
        co = float(np.corrcoef(unc, err)[0, 1])
        res[name] = dict(aurg=round(a, 4), corr_with_error=round(co, 3))
        curves[name] = c
        print(f"  {name:<26} AURG {a:+.4f} | corr {co:+.3f}")
    # poorest-decile capture: of the worst-error 20%, what frac are flagged by darkness vs ens σ?
    worst = err >= np.quantile(err, 0.80)
    for name, unc in [("−NL", -nl), ("ensemble σ", es), ("hetero σ", hs)]:
        flagged = unc >= np.quantile(unc, 0.80)
        rec = (worst & flagged).sum() / worst.sum()
        print(f"  flag the worst-20% errors via {name}: recall {rec:.0%}")
    json.dump(res, open("results/reliability_flag.json", "w"), indent=2)

    # figure: sparsification curves
    oracle = aurg(err, err, fracs)[1]
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    cols = {"Night-light darkness (−NL)": RED, "Deep ensemble σ": TEAL, "MC-dropout σ": AMBER, "Heteroscedastic σ": PLUM}
    for name, c in curves.items():
        lw = 3.0 if name.startswith("Night") else 1.8
        ax.plot(fracs, c, "-", color=cols[name], lw=lw, label=f"{name} (AURG {res[name]['aurg']:+.3f})")
    ax.plot(fracs, oracle, "--", color=NAVY, lw=1.6, label="oracle")
    ax.axhline(1.0, ls=":", color=GREY, lw=1.4, label="random")
    ax.set_xlabel("fraction of least-reliable villages removed")
    ax.set_ylabel("RMSE of remainder (÷ full)")
    ax.set_title("A trivial 'is it dark?' flag beats learned uncertainty at finding bad predictions",
                 color=NAVY, fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9); ax.grid(True, color="#eee")
    fig.tight_layout(); fig.savefig(OUT/"reliability_flag.png", dpi=170, bbox_inches="tight", facecolor="white")
    print("\nwrote results/reliability_flag.json + reliability_flag.png")


if __name__ == "__main__":
    main()
