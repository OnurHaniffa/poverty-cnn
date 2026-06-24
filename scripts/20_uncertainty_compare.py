"""Compare the two uncertainty sources and isolate WHY uncertainty is (un)informative.

- Ensemble (3-seed disagreement) vs MC-dropout (30 passes) — do they agree, and
  does either predict error?
- WITHIN-STRATUM check: the systematic bias (regression to mean) varies across the
  wealth range and can MASK a real uncertainty signal. So we also compute
  corr(uncertainty, error) WITHIN each wealth decile (bias ~constant there) and
  average. If within-decile corr >> global corr, uncertainty captures the random
  component but is swamped by systematic bias globally.

Run on PC: PPY scripts/20_uncertainty_compare.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from poverty_cnn.data import splits

META="data/processed/tile_cache/cache_metadata.csv"
SEEDS=["results/cnn_stable","results/cnn_stable_s1","results/cnn_stable_s2"]
RUN="results/cnn_stable"


def ensemble(meta):
    parts=[]
    for fold in splits.fold_ids():
        rows=splits.clusters_for(meta,fold,"test"); sub=meta.iloc[rows]
        P=[]; y0=None
        for d in SEEDS:
            z=np.load(f"{d}/preds_fold{fold}.npz",allow_pickle=True)
            if y0 is None: y0=z["y"]
            P.append(z["pred"])
        P=np.stack(P)
        parts.append(pd.DataFrame({"y":y0,"mean":P.mean(0),"std":P.std(0)}))
    return pd.concat(parts,ignore_index=True)


def mcdrop(meta):
    parts=[]
    for fold in splits.fold_ids():
        z=np.load(f"{RUN}/mc_preds_fold{fold}.npz",allow_pickle=True)
        parts.append(pd.DataFrame({"y":z["y"],"mean":z["mc_mean"],"std":z["mc_std"]}))
    return pd.concat(parts,ignore_index=True)


def within_stratum_corr(df, k=10):
    df=df.copy(); df["dec"]=pd.qcut(df.y,k,labels=False)
    cs=[]
    for _,g in df.groupby("dec"):
        if len(g)>20 and g["std"].std()>1e-9:
            cs.append(np.corrcoef(g["std"], np.abs(g.y-g["mean"]))[0,1])
    return np.nanmean(cs)


def report(name, df):
    err=np.abs(df.y-df["mean"])
    glob=np.corrcoef(df["std"],err)[0,1]
    win=within_stratum_corr(df)
    order=df.assign(err=err).sort_values("std")
    n=len(df); mae10=order.iloc[:int(0.1*n)]["err"].mean(); mae100=order["err"].mean()
    print(f"  {name:10}: corr(unc,err) global {glob:+.3f} | within-decile {win:+.3f} | "
          f"MAE conf-10% {mae10:.3f} vs all {mae100:.3f}")
    return df["std"].values


def main():
    meta=pd.read_csv(META)
    ens=ensemble(meta); mc=mcdrop(meta)
    print("### uncertainty validation — does it predict error?")
    es=report("ensemble", ens)
    ms=report("MC-dropout", mc)
    print(f"\n  agreement corr(ensemble_std, mc_std) = {np.corrcoef(es,ms)[0,1]:+.3f}")
    print( "  --> if BOTH have ~0 global corr AND low within-decile corr, the dominant")
    print( "      error is systematic bias, invisible to uncertainty (the honest caution).")


if __name__ == "__main__":
    main()
