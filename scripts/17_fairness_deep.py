"""Deep fairness rigor — the checks that matter most for an aid-targeting tool.

A) Per-country urban/rural: does the rural-ranking gap hold WITHIN countries, or
   is the pooled gap a composition artifact? KEY confound test: does the gap
   correlate with rural homogeneity (rural less spread -> intrinsically harder)?
B) Performance across the true-wealth range: is the model worst for the POOREST
   villages (where targeting matters most)? MAE by wealth decile.
C) Calibration: does the model regress to the mean (compress predictions)? Slope
   of pred~true + bias by decile. Compression => poor villages look less poor
   => under-targeted. Direct policy implication.

Run on PC: PPY scripts/17_fairness_deep.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poverty_cnn.data import splits

try:
    from scipy.stats import spearmanr
    def spear(a, b):
        if len(a) < 5: return np.nan
        return float(spearmanr(a, b)[0])
except Exception:
    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; GREY="#9aa4ae"; RED="#c0504d"
META="data/processed/tile_cache/cache_metadata.csv"
SEEDS=["results/cnn_stable","results/cnn_stable_s1","results/cnn_stable_s2"]
OUT=Path("results/figures/fairness"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})


def load(d, meta):
    cs,us,ys,ps=[],[],[],[]
    for fold in splits.fold_ids():
        z=np.load(f"{d}/preds_fold{fold}.npz",allow_pickle=True)
        rows=splits.clusters_for(meta,fold,"test"); sub=meta.iloc[rows]
        cs.append(sub["country"].to_numpy());us.append(sub["urban"].to_numpy())
        ys.append(z["y"]);ps.append(z["pred"])
    return pd.DataFrame({"country":np.concatenate(cs),"urban":np.concatenate(us),
                         "y":np.concatenate(ys),"pred":np.concatenate(ps)})


def main():
    meta=pd.read_csv(META)
    dfs=[load(d,meta) for d in SEEDS]
    pooled=pd.concat(dfs, ignore_index=True)   # 3 seeds stacked, for calibration

    # ===== A) per-country urban/rural =====
    print("### A) PER-COUNTRY urban vs rural ranking (3-seed mean Spearman), min 30/group")
    cc=sorted(dfs[0]["country"].unique())
    recs=[]
    for c in cc:
        us, rs, ustds, rstds = [], [], [], []
        for df in dfs:
            cu=df[(df.country==c)&(df.urban=="U")]; cr=df[(df.country==c)&(df.urban=="R")]
            if len(cu)>=30 and len(cr)>=30:
                us.append(spear(cu.y.values,cu.pred.values)); rs.append(spear(cr.y.values,cr.pred.values))
        if not us: continue
        cu0=dfs[0][(dfs[0].country==c)&(dfs[0].urban=="U")]; cr0=dfs[0][(dfs[0].country==c)&(dfs[0].urban=="R")]
        recs.append(dict(cc=c, u=np.nanmean(us), r=np.nanmean(rs),
                         nU=len(cu0), nR=len(cr0),
                         homog=cu0.y.std()/ (cr0.y.std()+1e-9)))
    A=pd.DataFrame(recs); A["gap"]=A["u"]-A["r"]
    n_urban_better=(A["gap"]>0).sum()
    print(f"  countries with enough of both: {len(A)}")
    print(f"  urban ranks better in {n_urban_better}/{len(A)} countries; median gap {A['gap'].median():+.3f}")
    corr=np.corrcoef(A["gap"], A["homog"])[0,1]
    print(f"  CONFOUND TEST: corr(urban-rural gap, urban/rural wealth-std ratio) = {corr:+.3f}")
    print( "    ~0 => gap is NOT explained by rural homogeneity (genuine model effect)")
    print( "    >0 => gap partly an artifact of rural being more homogeneous")

    fig,ax=plt.subplots(figsize=(7,7))
    As=A.sort_values("r")
    y=np.arange(len(As))
    ax.hlines(y, As["r"], As["u"], color=GREY, lw=1, zorder=1)
    ax.scatter(As["r"], y, color=AMBER, label="rural", zorder=3)
    ax.scatter(As["u"], y, color=TEAL, label="urban", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(As["cc"]); ax.set_xlabel("Spearman (ranking) — urban vs rural, per country")
    ax.legend(frameon=False); ax.set_title(f"Urban vs rural ranking per country\nurban better in {n_urban_better}/{len(A)}; gap~homogeneity r={corr:+.2f}")
    fig.tight_layout(); fig.savefig(OUT/"04_percountry_urbanrural.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ===== B/C) calibration + performance across wealth range =====
    y=pooled.y.values; p=pooled.pred.values
    slope, intc = np.polyfit(y, p, 1)
    print(f"\n### C) CALIBRATION: pred ~ {slope:.2f}*true + {intc:+.2f}")
    print(f"  slope {slope:.2f} (<1 = regression to mean / compression). pred std/true std = {p.std()/y.std():.2f}")

    pooled2=pooled.copy(); pooled2["dec"]=pd.qcut(pooled2.y, 10, labels=False)
    g=pooled2.groupby("dec").apply(lambda d: pd.Series({
        "true_mean":d.y.mean(),"pred_mean":d.pred.mean(),
        "bias":(d.pred-d.y).mean(),"mae":(d.pred-d.y).abs().mean()}))
    print("\n### B) by true-wealth decile (0=poorest):")
    print("  dec true_mean pred_mean  bias    MAE")
    for i,row in g.iterrows():
        print(f"   {i}  {row.true_mean:+.2f}    {row.pred_mean:+.2f}   {row.bias:+.3f}  {row.mae:.3f}")

    # Fig 5: calibration
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(y[::13], p[::13], s=4, alpha=0.15, color=NAVY, zorder=2)
    ax.plot(g.true_mean, g.pred_mean, "-o", color=AMBER, zorder=4, label="binned mean")
    lim=[-2.2,3]; ax.plot(lim,lim,"--",color=GREY,label="perfect"); ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel("true wealth"); ax.set_ylabel("predicted wealth")
    ax.set_title(f"Calibration: slope={slope:.2f} (<1 = compresses toward mean)\npoorest predicted LESS poor than they are")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT/"05_calibration.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # Fig 6: bias + MAE by decile
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    x=g.index.values
    ax.bar(x-0.2, g.bias, width=0.4, color=RED, label="bias (pred-true)", zorder=3)
    ax.bar(x+0.2, g.mae, width=0.4, color=TEAL, label="MAE", zorder=3)
    ax.axhline(0,color="#888",lw=1); ax.set_xlabel("true-wealth decile (0=poorest, 9=richest)")
    ax.set_xticks(x); ax.legend(frameon=False)
    ax.set_title("Poorest villages: positive bias (predicted too rich) — under-targeted")
    fig.tight_layout(); fig.savefig(OUT/"06_bias_by_wealth.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    print("\nwrote 04_percountry_urbanrural, 05_calibration, 06_bias_by_wealth ->", OUT)


if __name__ == "__main__":
    main()
