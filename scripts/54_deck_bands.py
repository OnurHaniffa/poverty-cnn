"""Render the 8 channels grouped into meaningful 'senses' for the deck S9 slide."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
M=pd.read_csv("data/processed/tile_cache_full/cache_metadata.csv")
cache=np.load("data/processed/tile_cache_full/cache.npy", mmap_mode="r")
cand=M[(M.urban=="U")&(M.nan_frac<0.005)].sort_values("wealth_index_mean",ascending=False)
idx=int(cand.iloc[8].name)
t=np.asarray(cache[idx]).astype(float)
print("tile", t.shape, "country", M.loc[idx,'country'], "wealth", round(M.loc[idx,'wealth_index_mean'],2))
OUT=Path("results/figures/deck_bands"); OUT.mkdir(parents=True,exist_ok=True)
def st(x,lo=2,hi=98):
    a,b=np.nanpercentile(x,[lo,hi]); return np.clip((np.nan_to_num(x,nan=a)-a)/(b-a+1e-9),0,1)
def save(arr,name,cmap=None):
    fig,ax=plt.subplots(figsize=(3,3)); ax.axis("off"); ax.imshow(arr,cmap=cmap,interpolation="nearest" if name=="nl.png" else "bilinear")
    fig.subplots_adjust(0,0,1,1); fig.savefig(OUT/name,dpi=160,facecolor="white"); plt.close()
rgb=np.dstack([st(t[0]),st(t[1]),st(t[2])])**0.85
save(rgb,"rgb.png")
save(st(t[3]),"nir.png",cmap="YlGn")
save(st(t[4]),"swir.png",cmap="copper")
save(st(t[6]),"thermal.png",cmap="inferno")
save(st(t[7]),"nl.png",cmap="inferno")
print("saved:",[p.name for p in OUT.glob('*.png')])
