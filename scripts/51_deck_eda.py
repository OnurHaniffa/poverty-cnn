"""Regenerate EDA figures in the DECK palette (light, cyan/teal/navy) for slide embedding."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
CY="#16b9d0"; TEAL="#1aa985"; NAVY="#0f2636"; INK="#33454f"; GREY="#9fb1bd"
OUT=Path("results/figures/deck_eda"); OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":12,
                     "text.color":NAVY,"axes.labelcolor":NAVY,"xtick.color":INK,"ytick.color":INK,
                     "axes.edgecolor":"#c4d2d9"})
W=pd.read_csv("data/processed/multiround_wealth_index_clusters.csv")
M=pd.read_csv("data/processed/tile_cache_full/cache_metadata.csv")

# 1) box: wealth by country
order=W.groupby("country").wealth_index_mean.median().sort_values().index
fig,ax=plt.subplots(figsize=(11.4,4.0))
bp=ax.boxplot([W[W.country==c].wealth_index_mean.values for c in order],patch_artist=True,
   showfliers=True,flierprops=dict(marker=".",ms=2.5,mfc=GREY,mec="none",alpha=.35),
   medianprops=dict(color=TEAL,lw=2.2),whiskerprops=dict(color="#90a4af"),capprops=dict(color="#90a4af"))
for b in bp["boxes"]: b.set(facecolor=CY,alpha=.5,edgecolor=NAVY,lw=1)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order,rotation=60,fontsize=9)
ax.set_ylabel("cluster wealth index"); ax.axhline(0,color=GREY,ls=":",lw=1)
fig.tight_layout(); fig.savefig(OUT/"box_country.png",dpi=200,facecolor="white"); plt.close()

# 2) box: urban vs rural
fig,ax=plt.subplots(figsize=(4.7,4.0))
bp=ax.boxplot([W[W.urban=="U"].wealth_index_mean.values,W[W.urban=="R"].wealth_index_mean.values],
   patch_artist=True,showfliers=True,flierprops=dict(marker=".",ms=2.5,mfc=GREY,mec="none",alpha=.35),
   medianprops=dict(color=NAVY,lw=2.2),whiskerprops=dict(color="#90a4af"),capprops=dict(color="#90a4af"))
for b,c in zip(bp["boxes"],[CY,TEAL]): b.set(facecolor=c,alpha=.5,edgecolor=NAVY,lw=1)
ax.set_xticklabels(["Urban","Rural"],fontsize=13); ax.set_ylabel("cluster wealth index"); ax.axhline(0,color=GREY,ls=":",lw=1)
fig.tight_layout(); fig.savefig(OUT/"box_urbanrural.png",dpi=200,facecolor="white"); plt.close()

# 3) missingness histogram
fig,ax=plt.subplots(figsize=(7.0,3.7))
ax.hist(M.nan_frac*100,bins=40,color=CY,alpha=.8,edgecolor=NAVY,lw=.5)
ax.set_xlabel("% missing / cloud pixels per tile"); ax.set_ylabel("number of tiles")
fig.tight_layout(); fig.savefig(OUT/"missing_hist.png",dpi=200,facecolor="white"); plt.close()
print("OK deck_eda:", [p.name for p in OUT.glob('*.png')])
print(f"nan: mean {M.nan_frac.mean()*100:.2f}% | <5% {(M.nan_frac<0.05).mean()*100:.0f}% | max {M.nan_frac.max()*100:.1f}%")
print(f"urban median {W[W.urban=='U'].wealth_index_mean.median():.2f} | rural {W[W.urban=='R'].wealth_index_mean.median():.2f}")
