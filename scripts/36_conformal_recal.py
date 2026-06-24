"""P3 (conformal prediction) + P4 (post-hoc recalibration) — finishing the UQ contribution.

CPU only, on existing pooled out-of-sample predictions (pilot models: 5-seed ensemble +
heteroscedastic). Joins urban/rural + true wealth from the pilot cache metadata via the same
deterministic test-row order the dataset uses.

P3 SPLIT CONFORMAL (heteroscedastic, normalized scores  s_i = |y-mean|/sigma):
  - marginal split-conformal interval -> coverage ~ nominal 90% (distribution-free guarantee,
    under an i.i.d. calib/test split)
  - coverage BY WEALTH DECILE under that marginal interval -> the poorest are UNDER-covered
  - Mondrian (group-conditional) conformal by urban/rural -> equalized per-group coverage
  - country-BLOCKED split -> coverage degradation: our cross-country design breaks
    exchangeability, so the guarantee weakens. We REPORT that degradation as a finding
    (Barber et al. 2023), rather than hiding it.

P4 POST-HOC RECALIBRATION (deep ensemble — the badly-miscalibrated, overconfident one):
  - variance-scaling factor s = sqrt(mean((y-mean)^2/sigma^2)) fit on a calibration split
  - report ENCE / PICP@90 / NLL pre vs post (marginal calibration is fixed)
  - coverage BY WEALTH DECILE pre vs post -> the poorest stay under-covered, because uniform
    scaling fixes the LEVEL of the bars, not their wealth-dependent SHAPE. Equity gap survives
    recalibration -> reinforces "calibration != equity".

Run on PC (CPU): PPY scripts/36_conformal_recal.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from poverty_cnn.data import splits

META = "data/processed/tile_cache/cache_metadata.csv"
FOLDS = ["A", "B", "C", "D", "E"]
ENS = ["results/cnn_stable", "results/cnn_stable_s1", "results/cnn_stable_s2",
       "results/cnn_stable_s3", "results/cnn_stable_s4"]
HET = "results/cnn_hetero"
NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; RED="#c0504d"; GREY="#9aa4ae"
OUT = Path("results/figures/uncertainty"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})


def load():
    meta = pd.read_csv(META)
    parts = []
    for f in FOLDS:
        het = np.load(f"{HET}/preds_fold{f}.npz", allow_pickle=True)
        ens_preds = np.stack([np.load(f"{r}/preds_fold{f}.npz")["pred"] for r in ENS])
        y = het["y"]
        rows = splits.clusters_for(meta, f, "test"); sub = meta.iloc[rows]
        assert len(sub) == len(y), f"row mismatch fold {f}"
        parts.append(pd.DataFrame({
            "fold": f, "country": sub.country.to_numpy(), "urban": sub.urban.to_numpy(), "y": y,
            "het_mean": het["mean"], "het_sigma": het["sigma"],
            "ens_mean": ens_preds.mean(0), "ens_sigma": ens_preds.std(0, ddof=1)}))
    return pd.concat(parts, ignore_index=True)


def conf_q(scores, alpha=0.10):
    n = len(scores)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)     # finite-sample correction
    return float(np.quantile(scores, lvl, method="higher"))


def cov_by_decile(y, covered, q=10):
    dec = pd.qcut(y, q, labels=False)
    return [float(covered[dec == d].mean()) for d in range(q)]


def gauss_metrics(y, mean, sigma):
    z = (y - mean) / np.clip(sigma, 1e-9, None)
    nll = float(np.mean(0.5*np.log(2*np.pi*sigma**2) + 0.5*z**2))
    lv = np.linspace(0.05, 0.95, 19)
    emp = np.array([np.mean(np.abs(z) <= norm.ppf(0.5+p/2)) for p in lv])
    ence = float(np.mean(np.abs(emp - lv)))
    picp90 = float(np.mean(np.abs(z) <= norm.ppf(0.95)))
    return dict(nll=nll, ence=ence, picp90=picp90)


def main():
    df = load()
    rng = np.random.default_rng(0)
    out = {}
    print(f"pooled OOS rows: {len(df)} | countries: {df.country.nunique()}")

    # ===================== P3: SPLIT CONFORMAL (heteroscedastic) =====================
    idx = rng.permutation(len(df)); half = len(df)//2
    cal, te = df.iloc[idx[:half]].copy(), df.iloc[idx[half:]].copy()
    s_cal = (np.abs(cal.y - cal.het_mean) / cal.het_sigma).values
    q = conf_q(s_cal, 0.10)
    te_cov = (np.abs(te.y - te.het_mean).values <= q * te.het_sigma.values)
    marg_cov = float(te_cov.mean()); marg_w = float((2*q*te.het_sigma).mean())
    dec_cov = cov_by_decile(te.y.values, te_cov)
    print("\n### P3 split-conformal (i.i.d. split, target 90%)")
    print(f"  marginal coverage {marg_cov:.3f} | mean width {marg_w:.2f}")
    print(f"  coverage by wealth decile (0=poorest): "
          + " ".join(f"{c:.2f}" for c in dec_cov))
    print(f"  -> poorest-decile {dec_cov[0]:.2f} vs richest-decile {dec_cov[-1]:.2f} "
          f"(gap {dec_cov[-1]-dec_cov[0]:+.2f})")

    # Mondrian (group-conditional) conformal by urban/rural
    mond = {}
    te_cov_m = np.zeros(len(te), dtype=bool)
    for g in ["U", "R"]:
        cg = cal[cal.urban == g]; tgmask = (te.urban == g).values
        qg = conf_q((np.abs(cg.y - cg.het_mean)/cg.het_sigma).values, 0.10)
        tg = te[te.urban == g]
        cov_g = (np.abs(tg.y - tg.het_mean).values <= qg * tg.het_sigma.values)
        mond[g] = dict(q=qg, coverage=float(cov_g.mean()), n=int(len(tg)))
        te_cov_m[tgmask] = cov_g
    print(f"  marginal-q coverage by group:   urban {float(te_cov[(te.urban=='U').values].mean()):.3f} | "
          f"rural {float(te_cov[(te.urban=='R').values].mean()):.3f}")
    print(f"  Mondrian (per-group q) coverage: urban {mond['U']['coverage']:.3f} | rural {mond['R']['coverage']:.3f}  (equalized)")

    # country-blocked split (exchangeability broken)
    countries = np.sort(df.country.unique())
    cset = set(countries[::2])                              # alternate countries -> calib
    calc = df[df.country.isin(cset)]; tec = df[~df.country.isin(cset)]
    qc = conf_q((np.abs(calc.y - calc.het_mean)/calc.het_sigma).values, 0.10)
    cov_block = float((np.abs(tec.y - tec.het_mean).values <= qc*tec.het_sigma.values).mean())
    print(f"  country-BLOCKED split coverage {cov_block:.3f}  (vs 0.90 target; gap = exchangeability cost)")

    out["conformal"] = dict(marginal_coverage=marg_cov, mean_width=marg_w,
                            coverage_by_decile=dec_cov, mondrian=mond,
                            country_blocked_coverage=cov_block)

    # ===================== P4: RECALIBRATION (deep ensemble) =====================
    pre = gauss_metrics(te.y.values, te.ens_mean.values, te.ens_sigma.values)
    z = (cal.y - cal.ens_mean).values / np.clip(cal.ens_sigma.values, 1e-9, None)
    s = float(np.sqrt(np.mean(z**2)))                      # variance-scaling factor
    post = gauss_metrics(te.y.values, te.ens_mean.values, s*te.ens_sigma.values)
    # coverage @90 by decile, pre vs post
    z90 = norm.ppf(0.95)
    cov_pre = np.abs(te.y - te.ens_mean).values <= z90*te.ens_sigma.values
    cov_post = np.abs(te.y - te.ens_mean).values <= z90*s*te.ens_sigma.values
    dec_pre, dec_post = cov_by_decile(te.y.values, cov_pre), cov_by_decile(te.y.values, cov_post)
    print(f"\n### P4 variance-scaling recalibration of deep ensemble (s={s:.2f})")
    print(f"  ENCE  {pre['ence']:.3f} -> {post['ence']:.3f}")
    print(f"  PICP90 {pre['picp90']:.3f} -> {post['picp90']:.3f}   (target 0.90)")
    print(f"  NLL   {pre['nll']:.2f} -> {post['nll']:.2f}")
    print(f"  poorest-decile coverage@90  {dec_pre[0]:.2f} -> {dec_post[0]:.2f}  "
          f"(richest {dec_pre[-1]:.2f} -> {dec_post[-1]:.2f})")
    print( "  -> recalibration fixes MARGINAL calibration but the poorest stay under-covered")
    out["recalibration"] = dict(scale_s=s, pre=pre, post=post,
                                cov_decile_pre=dec_pre, cov_decile_post=dec_post)

    # ---------- figures ----------
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhline(0.90, ls="--", color=GREY, label="target 90%")
    ax.plot(x, out["conformal"]["coverage_by_decile"], "-o", color=TEAL, lw=2, label="conformal coverage")
    ax.set_xlabel("true-wealth decile (0=poorest)"); ax.set_ylabel("coverage")
    ax.set_title("Conformal intervals: the poorest are under-covered", color=NAVY, fontweight="bold")
    ax.set_ylim(0, 1); ax.legend(frameon=False); ax.grid(True, color="#eee")
    fig.tight_layout(); fig.savefig(OUT/"conformal_coverage_by_wealth.png", dpi=170, bbox_inches="tight", facecolor="white")

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhline(0.90, ls="--", color=GREY, label="target 90%")
    ax.bar(x-0.2, dec_pre, width=0.4, color=GREY, label="ensemble (pre-recal)")
    ax.bar(x+0.2, dec_post, width=0.4, color=AMBER, label=f"recalibrated (×{s:.1f})")
    ax.set_xlabel("true-wealth decile (0=poorest)"); ax.set_ylabel("coverage@90%")
    ax.set_title("Recalibration lifts the LEVEL, not the poorest's gap", color=NAVY, fontweight="bold")
    ax.set_ylim(0, 1); ax.legend(frameon=False); ax.grid(True, color="#eee")
    fig.tight_layout(); fig.savefig(OUT/"recalibration_by_wealth.png", dpi=170, bbox_inches="tight", facecolor="white")

    json.dump(out, open("results/conformal_recal.json", "w"), indent=2)
    print("\nwrote results/conformal_recal.json + conformal_coverage_by_wealth.png + recalibration_by_wealth.png")


if __name__ == "__main__":
    main()
