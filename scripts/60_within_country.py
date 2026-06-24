import numpy as np, pandas as pd, glob
from scipy.stats import spearmanr, pearsonr
Y,P,C=[],[],[]
for f in sorted(glob.glob("results/cnn_stable/preds_fold*.npz")):
    z=np.load(f,allow_pickle=True);Y+=list(z['y']);P+=list(z['pred']);C+=list(z['country'])
df=pd.DataFrame({'y':np.array(Y,float),'p':np.array(P,float),'c':[str(x) for x in C]})
# pooled
sp_pool=spearmanr(df.y,df.p)[0]; r2_pool=1-((df.y-df.p)**2).sum()/((df.y-df.y.mean())**2).sum()
# per-country (WITHIN-country: between-country signal removed by construction)
rows=[]
for c,g in df.groupby('c'):
    if len(g)<50: continue
    sp=spearmanr(g.y,g.p)[0]; slope=np.polyfit(g.y,g.p,1)[0]
    k=max(int(len(g)*0.2),5); pi=g.y.values.argsort()[:k]
    pbias=(g.p.values[pi]-g.y.values[pi]).mean()
    # urban/rural would need the tag; skip here
    rows.append((c,len(g),sp,slope,pbias))
R=pd.DataFrame(rows,columns=['c','n','sp','slope','pbias'])
w=R.n/R.n.sum()
print("=== WITHIN-COUNTRY ROBUSTNESS (between-country signal removed) ===")
print(f"POOLED        Spearman {sp_pool:.2f}   r2 {r2_pool:.2f}")
print(f"WITHIN-country (n-weighted mean across {len(R)} countries):")
print(f"  Spearman          {(R.sp*w).sum():.2f}   (range {R.sp.min():.2f}-{R.sp.max():.2f})")
print(f"  pred~true slope    {(R.slope*w).sum():.2f}   (1.0=no shrinkage; <1 = regression to mean)")
print(f"  poorest-20% bias  {(R.pbias*w).sum():+.2f}   (>0 = over-predicts the within-country poorest)")
print(f"  countries where slope<1 (shrinkage): {(R.slope<1).sum()}/{len(R)}")
print(f"  countries where poorest over-predicted (bias>0): {(R.pbias>0).sum()}/{len(R)}")
# is per-household asset data available for full per-country PCA?
import os
print("=== household asset files for per-country PCA ? ===")
for p in glob.glob("data/processed/*asset*")+glob.glob("data/processed/*household*")+glob.glob("data/processed/*hh*"): print(" ",p)
