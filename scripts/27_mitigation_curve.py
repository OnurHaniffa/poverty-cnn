"""Mitigation trade-off curve: the equity-accuracy frontier of reweighting.

For the baseline + each reweight strength alpha (fold A), compute overall r2,
poorest-decile bias (predicted-too-rich; 0 = unbiased), and poorest-20% recall.
Plots them vs alpha so we can pick the sweet spot and SHOW the trade-off.

Run on PC: PPY scripts/27_mitigation_curve.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; RED="#c0504d"
OUT=Path("results/figures/mitigation"); OUT.mkdir(parents=True, exist_ok=True)

# (alpha, run-dir) — baseline is alpha 0
RUNS=[(0.0,"results/cnn_stable"),(0.5,"results/cnn_reweighted"),
      (1.0,"results/cnn_reweighted_a10"),(1.5,"results/cnn_reweighted_a15")]


def metrics(d):
    z=np.load(f"{d}/preds_foldA.npz"); y,p=z["y"],z["pred"]
    r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    dec0=y<=np.quantile(y,0.10); bias0=float((p[dec0]-y[dec0]).mean())
    nt=int(len(y)*0.20); rec=len(set(np.argsort(y)[:nt])&set(np.argsort(p)[:nt]))/nt
    return float(r2),bias0,float(rec)


def main():
    rows=[]
    for a,d in RUNS:
        if not Path(f"{d}/preds_foldA.npz").exists():
            print(f"  (missing {d} — skip)"); continue
        r2,b,rec=metrics(d); rows.append((a,r2,b,rec))
        print(f"  alpha {a}: r2 {r2:+.3f} | poorest-decile bias {b:+.3f} | poorest-20% recall {rec:.1%}")
    if len(rows)<2: print("not enough runs yet"); return
    a=[r[0] for r in rows]; r2=[r[1] for r in rows]; bias=[r[2] for r in rows]; rec=[r[3] for r in rows]

    fig,ax=plt.subplots(1,3,figsize=(12,4))
    ax[0].plot(a,bias,"-o",color=RED); ax[0].axhline(0,color="#aaa",ls="--")
    ax[0].set_title("Poorest-decile bias\n(toward 0 = fixed)"); ax[0].set_xlabel("reweight strength alpha")
    ax[1].plot(a,r2,"-o",color=NAVY); ax[1].set_title("Overall r2\n(cost of reweighting)"); ax[1].set_xlabel("alpha")
    ax[2].plot(a,[x*100 for x in rec],"-o",color=TEAL); ax[2].set_title("Poorest-20% recall %\n(targeting)"); ax[2].set_xlabel("alpha")
    for x in ax: x.grid(True,color="#eee")
    fig.suptitle("Mitigation trade-off: reweighting cuts the poorest-bias — at what cost? (fold A)",y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"01_tradeoff_curve.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("\nwrote",OUT/"01_tradeoff_curve.png")


if __name__=="__main__":
    main()
