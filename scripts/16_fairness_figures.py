"""Fairness-audit figures (the honest versions, per the verification in 15).

Fig 1: urban vs rural across r2/MAE/Spearman (3-seed mean +/- std). Tells the
       'equal absolute accuracy, worse rural ranking' story (MAE bars ~equal).
Fig 2: per-country r2 with the across-seed noise floor (error bars) + class colors.
Fig 3: NL-only vs MS+NL per-country r2 (complementarity / the flips).

Run on PC: PPY scripts/16_fairness_figures.py
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
    def spear(a, b): return float(spearmanr(a, b)[0])
except Exception:
    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

def r2(a, b): return 1.0 - ((a - b) ** 2).sum() / (((a - a.mean()) ** 2).sum() + 1e-12)
def mae(a, b): return float(np.abs(a - b).mean())

NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; GREY="#9aa4ae"; RED="#c0504d"
META="data/processed/tile_cache/cache_metadata.csv"
SEEDS={"s42":"results/cnn_stable","s1":"results/cnn_stable_s1","s2":"results/cnn_stable_s2"}
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

def met(df,k,v,fn):
    m=df[df[k]==v]; return fn(m["y"].to_numpy(),m["pred"].to_numpy())


def main():
    meta=pd.read_csv(META)
    seed_dfs={s:load(d,meta) for s,d in SEEDS.items()}
    dfs=list(seed_dfs.values())

    # ---- Fig 1: urban vs rural, 3 metrics, 3-seed mean+/-std ----
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,(fn,name,lo,hi) in zip(axes,[(r2,"r²",0,0.4),(mae,"MAE (lower=better)",0,0.55),(spear,"Spearman",0,0.7)]):
        means=[]; stds=[]
        for g in ["U","R"]:
            v=[met(df,"urban",g,fn) for df in dfs]; means.append(np.mean(v)); stds.append(np.std(v))
        ax.bar(["urban","rural"],means,yerr=stds,color=[TEAL,AMBER],capsize=5,width=0.6,zorder=3)
        ax.set_title(name); ax.set_ylim(lo,hi); ax.yaxis.grid(True,color="#eee",zorder=0)
    fig.suptitle("Urban vs Rural — equal absolute accuracy (MAE), worse rural ranking (Spearman)",
                 fontsize=12,y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"01_urban_rural.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ---- Fig 2: per-country r2 with noise floor + class ----
    cc=sorted(seed_dfs["s42"]["country"].unique())
    rows=[]
    for c in cc:
        r2v=[met(df,"country",c,r2) for df in dfs]; spv=np.mean([met(df,"country",c,spear) for df in dfs])
        mav=np.mean([met(df,"country",c,mae) for df in dfs])
        ystd=seed_dfs["s42"][seed_dfs["s42"]["country"]==c]["y"].std()
        rows.append((c,np.mean(r2v),np.std(r2v),mav,spv,ystd))
    rows.sort(key=lambda x:x[1])
    def cls(rm,mm,sm,ys):
        if rm>=0.45: return "well-served",TEAL
        if sm>=0.70: return "miscalibrated",AMBER
        if ys<0.65 and mm<0.40: return "low-variance",GREY
        return "genuine miss",RED
    cols=[cls(rm,mm,sm,ys)[1] for _,rm,_,mm,sm,ys in rows]
    fig,ax=plt.subplots(figsize=(7,7.5))
    ax.barh([r[0] for r in rows],[r[1] for r in rows],xerr=[r[2] for r in rows],
            color=cols,capsize=3,zorder=3)
    ax.axvline(0,color="#888",lw=1); ax.set_xlabel("per-country r²  (3-seed mean ± std = noise floor)")
    ax.set_title("Who the model serves — per-country, with training-noise error bars")
    from matplotlib.patches import Patch
    leg=[Patch(color=TEAL,label="well-served"),Patch(color=AMBER,label="miscalibrated (ranks OK)"),
         Patch(color=RED,label="genuine miss"),Patch(color=GREY,label="low-variance")]
    ax.legend(handles=leg,fontsize=9,loc="lower right",frameon=False)
    ax.xaxis.grid(True,color="#eee",zorder=0)
    fig.tight_layout(); fig.savefig(OUT/"02_percountry.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # ---- Fig 3: NL-only vs MS+NL per-country r2 (complementarity) ----
    full=seed_dfs["s42"]; nl=load("results/cnn_nl_stable",meta)
    fr=[met(full,"country",c,r2) for c in cc]; nr=[met(nl,"country",c,r2) for c in cc]
    fig,ax=plt.subplots(figsize=(6.5,6.5))
    ax.scatter(fr,nr,color=NAVY,zorder=3)
    for c,x,y in zip(cc,fr,nr):
        if abs(x-y)>0.25: ax.annotate(c,(x,y),fontsize=8,color=RED,xytext=(3,3),textcoords="offset points")
    lim=[-0.4,0.9]; ax.plot(lim,lim,"--",color=GREY,lw=1); ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel("MS+NL (full) per-country r²"); ax.set_ylabel("NL-only per-country r²")
    ax.set_title("Channels are COMPLEMENTARY per country (not redundant)\nr=%.2f — points far off diagonal = big flips"%np.corrcoef(fr,nr)[0,1])
    fig.tight_layout(); fig.savefig(OUT/"03_nl_vs_msnl.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    print("wrote 3 figures ->", OUT)


if __name__ == "__main__":
    main()
