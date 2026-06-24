"""EDA extras for the professor: box plots, band distributions, missingness, correlation."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
TEAL="#226E8C"; CYAN="#16CEE2"; GREEN="#1BAE8B"; NAVY="#12344A"; INK="#2B3036"; GREY="#9aa4ae"
OUT=Path("results/figures/eda"); OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":11})
W=pd.read_csv("data/processed/multiround_wealth_index_clusters.csv")
M=pd.read_csv("data/processed/tile_cache_full/cache_metadata.csv")

# 1) box: wealth by country (sorted by median)
order=W.groupby("country").wealth_index_mean.median().sort_values().index
fig,ax=plt.subplots(figsize=(11,4.6))
data=[W[W.country==c].wealth_index_mean.values for c in order]
bp=ax.boxplot(data,patch_artist=True,showfliers=True,flierprops=dict(marker=".",ms=3,mfc=GREY,mec="none",alpha=.4))
for b in bp["boxes"]: b.set(facecolor=TEAL,alpha=.65,edgecolor=NAVY)
for w in bp["medians"]: w.set(color=CYAN,lw=2)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order,rotation=60,fontsize=8.5)
ax.set_ylabel("cluster wealth index"); ax.set_title("Wealth distribution by country (box plot)",color=NAVY,fontweight="bold")
ax.axhline(0,color=GREY,ls=":",lw=1); fig.tight_layout(); fig.savefig(OUT/"box_wealth_by_country.png",dpi=160,facecolor="white"); plt.close()

# 2) box: wealth urban vs rural
fig,ax=plt.subplots(figsize=(5,4.6))
du=[W[W.urban=="U"].wealth_index_mean.values, W[W.urban=="R"].wealth_index_mean.values]
bp=ax.boxplot(du,patch_artist=True,showfliers=True,flierprops=dict(marker=".",ms=3,mfc=GREY,mec="none",alpha=.4))
for b,c in zip(bp["boxes"],[CYAN,GREEN]): b.set(facecolor=c,alpha=.6,edgecolor=NAVY)
for m in bp["medians"]: m.set(color=NAVY,lw=2)
ax.set_xticklabels(["Urban","Rural"]); ax.set_ylabel("cluster wealth index")
ax.set_title("Urban vs rural wealth (box plot)",color=NAVY,fontweight="bold"); ax.axhline(0,color=GREY,ls=":",lw=1)
fig.tight_layout(); fig.savefig(OUT/"box_wealth_urban_rural.png",dpi=160,facecolor="white"); plt.close()

# 3) per-band tile-mean distributions (sample) + 5) band correlation heatmap
cache=np.load("data/processed/tile_cache_full/cache.npy",mmap_mode="r")
rng=np.random.default_rng(0); idx=rng.choice(len(M),2500,replace=False)
bm=np.stack([np.asarray(cache[int(i)]).reshape(8,-1).mean(1) for i in idx])  # (2500,8)
names=["Red","Green","Blue","NIR","SWIR1","SWIR2","Thermal","Night-lights"]
z=(bm-bm.mean(0))/(bm.std(0)+1e-9)
fig,ax=plt.subplots(figsize=(9,4.4))
bp=ax.boxplot([z[:,k] for k in range(8)],patch_artist=True,showfliers=True,flierprops=dict(marker=".",ms=2,mfc=GREY,mec="none",alpha=.3))
for b in bp["boxes"]: b.set(facecolor=TEAL,alpha=.6,edgecolor=NAVY)
for m in bp["medians"]: m.set(color=CYAN,lw=2)
ax.set_xticklabels(names,rotation=30,fontsize=9); ax.set_ylabel("standardised tile-mean")
ax.set_title("Distribution of the 8 satellite channels (standardised)",color=NAVY,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"box_band_distributions.png",dpi=160,facecolor="white"); plt.close()

C=np.corrcoef(bm.T)
fig,ax=plt.subplots(figsize=(6,5.2)); im=ax.imshow(C,cmap="BrBG",vmin=-1,vmax=1)
ax.set_xticks(range(8)); ax.set_xticklabels(names,rotation=45,ha="right",fontsize=8.5); ax.set_yticks(range(8)); ax.set_yticklabels(names,fontsize=8.5)
for i in range(8):
    for j in range(8): ax.text(j,i,f"{C[i,j]:.1f}",ha="center",va="center",fontsize=7,color="black")
ax.set_title("Channel correlation",color=NAVY,fontweight="bold"); fig.colorbar(im,shrink=.8)
fig.tight_layout(); fig.savefig(OUT/"band_correlation.png",dpi=160,facecolor="white"); plt.close()

# 4) missing-data (nan_frac) distribution
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(M.nan_frac*100,bins=40,color=TEAL,alpha=.8,edgecolor=NAVY)
ax.set_xlabel("% missing/cloud pixels per tile"); ax.set_ylabel("# tiles")
ax.set_title(f"Tile data quality: {(M.nan_frac<0.05).mean()*100:.0f}% of tiles <5% missing",color=NAVY,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"missing_data.png",dpi=160,facecolor="white"); plt.close()
print("wrote 5 EDA figures:", [p.name for p in sorted(OUT.glob("box_*.png"))]+["band_correlation.png","missing_data.png"])
print(f"nan_frac: mean {M.nan_frac.mean()*100:.2f}% | max {M.nan_frac.max()*100:.1f}% | <5%: {(M.nan_frac<0.05).mean()*100:.0f}%")
