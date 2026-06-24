"""Final uncertainty synthesis: is heteroscedastic uncertainty (the only one with a
real positive corr) actually USEFUL, and does it know the model fails on the poorest?

Compares all 3 methods, then deep-dives heteroscedastic:
  - sigma vs error across the wealth range (does it flag the poorest, where ensemble
    was blind and the model is confidently wrong?)
  - sigma urban vs rural
  - risk-coverage (does abstaining on high-sigma lower error?)
Figures for the writeup.

Run on PC: PPY scripts/22_uncertainty_final.py
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


def hetero(meta):
    parts=[]
    for fold in splits.fold_ids():
        rows=splits.clusters_for(meta,fold,"test"); sub=meta.iloc[rows]
        z=np.load(f"results/cnn_hetero/preds_fold{fold}.npz",allow_pickle=True)
        parts.append(pd.DataFrame({"country":sub.country.to_numpy(),"urban":sub.urban.to_numpy(),
                                   "y":z["y"],"mean":z["mean"],"sigma":z["sigma"]}))
    df=pd.concat(parts,ignore_index=True); df["err"]=np.abs(df.y-df["mean"]); return df


def ens_std(meta):
    out=[]
    for fold in splits.fold_ids():
        P=[np.load(f"{d}/preds_fold{fold}.npz",allow_pickle=True)["pred"] for d in SEEDS]
        out.append(np.stack(P).std(0))
    return np.concatenate(out)


def mc_std(meta):
    return np.concatenate([np.load(f"results/cnn_stable/mc_preds_fold{f}.npz",allow_pickle=True)["mc_std"]
                           for f in splits.fold_ids()])


def main():
    meta=pd.read_csv(META)
    df=hetero(meta); n=len(df)

    # ---- Fig: 3-method validation ----
    corrs={"MC-dropout":np.corrcoef(mc_std(meta),df.err)[0,1],
           "Ensemble":np.corrcoef(ens_std(meta),df.err)[0,1],
           "Heteroscedastic":np.corrcoef(df.sigma,df.err)[0,1]}
    fig,ax=plt.subplots(figsize=(6,4))
    ks=list(corrs); ax.bar(ks,[corrs[k] for k in ks],color=[GREY,AMBER,TEAL],zorder=3)
    ax.axhline(0,color="#888",lw=1); ax.set_ylabel("corr(uncertainty, |error|)")
    ax.set_title("Only heteroscedastic (aleatoric) uncertainty\npredicts error — and only weakly")
    for i,k in enumerate(ks): ax.text(i,corrs[k]+0.005,f"{corrs[k]:+.3f}",ha="center")
    fig.tight_layout(); fig.savefig(OUT/"10_three_methods.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ---- Fig: does heteroscedastic sigma know the model fails on the poorest? ----
    df["dec"]=pd.qcut(df.y,10,labels=False)
    g=df.groupby("dec").agg(sigma=("sigma","mean"),err=("err","mean"))
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(g.index-0.2,g.sigma,0.4,color=TEAL,label="predicted sigma",zorder=3)
    ax.bar(g.index+0.2,g.err,0.4,color=RED,label="actual error (MAE)",zorder=3)
    ax.set_xlabel("true-wealth decile (0=poorest)"); ax.set_xticks(g.index); ax.legend(frameon=False)
    sc=np.corrcoef(g.sigma,g.err)[0,1]
    ax.set_title(f"Does heteroscedastic sigma track error across wealth? (decile corr {sc:+.2f})")
    fig.tight_layout(); fig.savefig(OUT/"11_hetero_by_wealth.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ---- Fig: risk-coverage (hetero vs ensemble) ----
    covs=np.linspace(0.1,1.0,10)
    he=df.sort_values("sigma").reset_index(drop=True)
    en=df.assign(es=ens_std(meta)).sort_values("es").reset_index(drop=True)
    hmae=[he.iloc[:int(c*n)].err.mean() for c in covs]; emae=[en.iloc[:int(c*n)].err.mean() for c in covs]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(covs*100,hmae,"-o",color=TEAL,label="heteroscedastic"); ax.plot(covs*100,emae,"-o",color=AMBER,label="ensemble")
    ax.set_xlabel("coverage % (most-confident kept)"); ax.set_ylabel("MAE"); ax.legend(frameon=False)
    ax.set_title("Risk-coverage: abstaining on high-sigma villages\n(downward-left = uncertainty helps)")
    ax.grid(True,color="#eee"); fig.tight_layout(); fig.savefig(OUT/"12_risk_coverage.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ---- numbers ----
    su=df[df.urban=="U"].sigma.mean(); sr=df[df.urban=="R"].sigma.mean()
    print("### heteroscedastic uncertainty — practical value")
    print(f"  corr(sigma,err) {corrs['Heteroscedastic']:+.3f}  vs ensemble {corrs['Ensemble']:+.3f}  vs MC {corrs['MC-dropout']:+.3f}")
    print(f"  sigma by wealth decile (poorest->richest): {[round(v,3) for v in g.sigma.values]}")
    print(f"  poorest-decile sigma {g.sigma.iloc[0]:.3f} vs richest {g.sigma.iloc[-1]:.3f}  "
          f"({'flags poorest' if g.sigma.iloc[0]>g.sigma.mean() else 'does NOT flag poorest'})")
    print(f"  sigma urban {su:.3f} vs rural {sr:.3f}")
    print(f"  risk-coverage: MAE conf-20% {hmae[1]:.3f} vs all {hmae[-1]:.3f}  "
          f"(ensemble {emae[1]:.3f} vs {emae[-1]:.3f})")
    print("\nwrote 10_three_methods, 11_hetero_by_wealth, 12_risk_coverage ->", OUT)


if __name__ == "__main__":
    main()
