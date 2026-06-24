"""Uncertainty (contribution #3) — Part 1: deep-ensemble uncertainty (the primary).

Per village, the 3 independently-trained seeds give 3 predictions (all from models
that never saw that village's country). Their disagreement (std) = epistemic
uncertainty. We then:
  A) VALIDATE — does higher uncertainty actually mean higher error? (the make-or-break
     check: correlation + risk-coverage curve). Useless uncertainty -> stop here.
  B) UNCERTAINTY-AWARE FAIRNESS — does the model KNOW where it fails? Is it more
     uncertain about the rural / poorest villages (where the audit showed it's worst)?
  C) SELECTIVE TARGETING — abstain on the most-uncertain (send to survey); does
     targeting accuracy improve on the confident subset, and who gets abstained?

Run on PC: PPY scripts/18_uncertainty.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poverty_cnn.data import splits

NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; GREY="#9aa4ae"; RED="#c0504d"
META="data/processed/tile_cache/cache_metadata.csv"
SEEDS=["results/cnn_stable","results/cnn_stable_s1","results/cnn_stable_s2"]
OUT=Path("results/figures/uncertainty"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})


def build(meta):
    """Aligned per-village df with ensemble mean + std across the 3 seeds."""
    parts=[]
    for fold in splits.fold_ids():
        rows=splits.clusters_for(meta,fold,"test"); sub=meta.iloc[rows]
        preds=[]
        y0=None
        for d in SEEDS:
            z=np.load(f"{d}/preds_fold{fold}.npz",allow_pickle=True)
            assert (sub["country"].to_numpy()==z["country"].astype(str)).all(), "alignment"
            if y0 is None: y0=z["y"]
            assert np.allclose(y0,z["y"]), "y mismatch across seeds"
            preds.append(z["pred"])
        P=np.stack(preds)                      # (3, n)
        parts.append(pd.DataFrame({"country":sub["country"].to_numpy(),
                                   "urban":sub["urban"].to_numpy(),
                                   "y":y0, "mean":P.mean(0), "std":P.std(0)}))
    df=pd.concat(parts, ignore_index=True)
    df["abs_err"]=np.abs(df["y"]-df["mean"])
    return df


def main():
    meta=pd.read_csv(META)
    df=build(meta)
    n=len(df)
    print(f"ensemble over {len(SEEDS)} seeds, {n} villages")
    print(f"uncertainty(std): mean {df['std'].mean():.3f}  range [{df['std'].min():.3f},{df['std'].max():.3f}]")

    # ===== A) VALIDATE =====
    c=np.corrcoef(df["std"],df["abs_err"])[0,1]
    print(f"\n### A) VALIDATE: corr(uncertainty, abs_error) = {c:+.3f}  (>0 = uncertainty predicts error)")
    # risk-coverage: keep most-confident coverage fraction, MAE on it
    order=df.sort_values("std").reset_index(drop=True)
    covs=np.linspace(0.1,1.0,10); maes=[order.iloc[:int(cv*n)]["abs_err"].mean() for cv in covs]
    print("  risk-coverage (MAE on most-confident fraction):")
    for cv,m in zip(covs,maes): print(f"    coverage {cv:.0%}: MAE {m:.3f}")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(covs*100,maes,"-o",color=TEAL); ax.set_xlabel("coverage % (most-confident kept)")
    ax.set_ylabel("MAE"); ax.set_title("Risk-coverage: abstaining on uncertain villages lowers error\n(upward slope = uncertainty is meaningful)")
    ax.grid(True,color="#eee"); fig.tight_layout(); fig.savefig(OUT/"07_risk_coverage.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ===== B) UNCERTAINTY-AWARE FAIRNESS =====
    print("\n### B) does uncertainty KNOW where the model fails?")
    su=df[df.urban=="U"]["std"].mean(); sr=df[df.urban=="R"]["std"].mean()
    print(f"  mean uncertainty: urban {su:.3f} | rural {sr:.3f}  ({'rural MORE uncertain' if sr>su else 'urban MORE uncertain'})")
    df["dec"]=pd.qcut(df.y,10,labels=False)
    g=df.groupby("dec").agg(std=("std","mean"),abs_err=("abs_err","mean"))
    print("  by true-wealth decile (0=poorest): std vs error")
    for i,row in g.iterrows(): print(f"    dec {i}: std {row['std']:.3f}  MAE {row['abs_err']:.3f}")
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(g.index-0.2,g["std"],0.4,color=AMBER,label="uncertainty (std)",zorder=3)
    ax.bar(g.index+0.2,g["abs_err"],0.4,color=RED,label="error (MAE)",zorder=3)
    ax.set_xlabel("true-wealth decile (0=poorest)"); ax.set_xticks(g.index); ax.legend(frameon=False)
    ax.set_title("Does uncertainty track error across the wealth range?")
    fig.tight_layout(); fig.savefig(OUT/"08_uncertainty_fairness.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ===== C) SELECTIVE TARGETING (within-country, abstain on uncertain) =====
    print("\n### C) selective targeting: within-country poorest-20%, abstain most-uncertain")
    def wc_recall(d,frac=0.20):
        out=[]
        for cc in d.country.unique():
            s=d[d.country==cc]; nt=int(len(s)*frac)
            if nt<3: continue
            out.append(len(set(np.argsort(s.y.values)[:nt])&set(np.argsort(s["mean"].values)[:nt]))/nt)
        return np.mean(out)
    covs2=[1.0,0.8,0.6,0.4]; recs=[]
    for cv in covs2:
        keep=df.sort_values("std").iloc[:int(cv*n)]   # most-confident
        recs.append(wc_recall(keep))
    for cv,rc in zip(covs2,recs): print(f"  coverage {cv:.0%} (trust model on most-confident): within-country poorest-20% recall {rc:.1%}")
    # who gets abstained? urban/rural share of the most-UNCERTAIN 30%
    unc=df.sort_values("std",ascending=False).iloc[:int(0.3*n)]
    print(f"  most-uncertain 30% -> rural share {100*(unc.urban=='R').mean():.0f}% (baseline rural {100*(df.urban=='R').mean():.0f}%)")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([c*100 for c in covs2],[r*100 for r in recs],"-o",color=NAVY)
    ax.set_xlabel("coverage % (trust model on most-confident)"); ax.set_ylabel("within-country poorest-20% recall %")
    ax.set_title("Selective targeting: trusting only confident predictions\nimproves who we catch")
    ax.grid(True,color="#eee"); fig.tight_layout(); fig.savefig(OUT/"09_selective_targeting.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    df.to_csv("results/uncertainty_ensemble.csv",index=False)
    print("\nwrote figures -> results/figures/uncertainty/ + results/uncertainty_ensemble.csv")


if __name__ == "__main__":
    main()
