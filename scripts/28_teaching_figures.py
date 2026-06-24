"""Teaching figures for the lab presentation (brand-styled, Avenir Next).

1. metrics_example.png — 5 villages, true vs predicted, making MAE / Spearman / r2
   intuitive. Chosen so predictions are imperfect (MAE) yet perfectly ordered
   (Spearman=1) — the 'ranking is what matters for targeting' lesson, and it also
   previews our regression-to-mean (predictions compressed toward the middle).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; SLATE="#2b3440"; GREY="#9aa4ae"; GRID="#e6ebf0"
plt.rcParams.update({"font.family":"Avenir Next","font.size":13,"text.color":SLATE,
    "axes.edgecolor":"#c7d0d9","axes.labelcolor":SLATE,"xtick.color":SLATE,"ytick.color":SLATE,
    "axes.spines.top":False,"axes.spines.right":False})
OUT=Path("results/figures/teaching"); OUT.mkdir(parents=True, exist_ok=True)

villages=["A","B","C","D","E"]
true=np.array([-1.5,-0.5,0.0,0.8,1.5])
pred=np.array([-0.8,-0.7,0.3,0.6,1.0])     # imperfect + compressed toward mean
mae=np.abs(true-pred).mean()
r2=1-((true-pred)**2).sum()/((true-true.mean())**2).sum()
rho=spearmanr(true,pred)[0]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.6,5.0),gridspec_kw={"width_ratios":[1.25,1]})

# --- Panel 1: how far off (MAE) ---
x=np.arange(5)
for i in x:
    ax1.plot([i,i],[true[i],pred[i]],color=GREY,lw=2,zorder=1)
ax1.scatter(x,true,s=150,color=TEAL,zorder=3,label="true wealth")
ax1.scatter(x,pred,s=150,color=AMBER,zorder=3,label="model's prediction")
for i in x:
    gap=abs(true[i]-pred[i]); ymid=(true[i]+pred[i])/2
    ax1.annotate(f"{gap:.1f}",(i,ymid),xytext=(7,0),textcoords="offset points",
                 color=GREY,fontsize=11,va="center")
ax1.axhline(0,color="#d8dee6",lw=1,zorder=0)
ax1.set_xticks(x); ax1.set_xticklabels([f"village {v}" for v in villages],fontsize=12)
ax1.set_ylabel("wealth index"); ax1.set_ylim(-2.1,2.1)
ax1.legend(frameon=False,fontsize=12,loc="upper left")
ax1.set_title("How far off?  $\\rightarrow$  MAE = %.2f"%mae,fontsize=15,color=NAVY,fontweight="bold",pad=12)
ax1.text(0.5,-1.95,"MAE = average of the gaps (in wealth units)",fontsize=11,color=GREY)

# --- Panel 2: right order (Spearman) ---
tr=np.argsort(np.argsort(true))+1; pr=np.argsort(np.argsort(pred))+1
ax2.plot([0.5,5.5],[0.5,5.5],"--",color=GREY,lw=1.5,zorder=1)
ax2.scatter(tr,pr,s=160,color=NAVY,zorder=3)
for i in x:
    ax2.annotate(villages[i],(tr[i],pr[i]),xytext=(8,-4),textcoords="offset points",color=NAVY,fontsize=12,fontweight="bold")
ax2.set_xlabel("true rank (poorest to richest)"); ax2.set_ylabel("predicted rank")
ax2.set_xticks(range(1,6)); ax2.set_yticks(range(1,6)); ax2.set_xlim(0.4,5.7); ax2.set_ylim(0.4,5.7)
ax2.set_title("Right order?  $\\rightarrow$  Spearman = %.2f"%rho,fontsize=15,color=NAVY,fontweight="bold",pad=12)
ax2.text(0.6,5.4,"on the line = order perfectly preserved",fontsize=11,color=GREY)

fig.suptitle("Same 5 villages: predictions are imperfect (MAE 0.38) — but perfectly ordered (Spearman 1.00)",
             fontsize=15.5,color=SLATE,y=1.02)
fig.text(0.5,-0.04,"r² = %.2f  ·  For finding the poorest, ORDER is what matters — a model can rank perfectly even when every value is a little off."%r2,
         ha="center",fontsize=12.5,color=TEAL,fontweight="bold")
fig.tight_layout()
fig.savefig(OUT/"metrics_example.png",dpi=200,bbox_inches="tight",facecolor="white")
print("wrote",OUT/"metrics_example.png","| MAE %.2f r2 %.2f rho %.2f"%(mae,r2,rho))
