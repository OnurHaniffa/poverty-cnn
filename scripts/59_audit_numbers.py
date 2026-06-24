import numpy as np, pandas as pd, glob, json, os
from scipy.stats import pearsonr, spearmanr
# ---- 1. country-bootstrap CIs on leave-country-out preds ----
Y=[];P=[];C=[]
for f in sorted(glob.glob("results/cnn_stable/preds_fold*.npz")):
    z=np.load(f,allow_pickle=True); Y+=list(z['y']);P+=list(z['pred']);C+=list(z['country'])
Y=np.array(Y,float);P=np.array(P,float);C=np.array([str(x) for x in C])
def met(idx):
    y,p=Y[idx],P[idx]; r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    return r2,pearsonr(y,p)[0],spearmanr(y,p)[0]
co=np.unique(C); rng=np.random.default_rng(0); B={'r2':[],'r':[],'sp':[]}
for _ in range(1000):
    cs=rng.choice(co,len(co),replace=True); idx=np.concatenate([np.where(C==c)[0] for c in cs])
    r2,r,sp=met(idx); B['r2'].append(r2);B['r'].append(r);B['sp'].append(sp)
r2,r,sp=met(np.arange(len(Y)))
print("=== HEADLINE (leave-country-out, country-bootstrap 95% CI) ===")
print(f"r2 = {r2:.2f}  CI[{np.percentile(B['r2'],2.5):.2f}, {np.percentile(B['r2'],97.5):.2f}]")
print(f"Pearson r = {r:.2f}  CI[{np.percentile(B['r'],2.5):.2f}, {np.percentile(B['r'],97.5):.2f}]")
print(f"Spearman = {sp:.2f}  CI[{np.percentile(B['sp'],2.5):.2f}, {np.percentile(B['sp'],97.5):.2f}]")
# poorest-decile bias CI (village bootstrap)
dec=pd.qcut(Y,10,labels=False); pm=dec==0
bb=[ (P[pm][i]-Y[pm][i]) for i in range(pm.sum())]
biasboot=[np.mean(rng.choice(np.array(bb),len(bb),replace=True)) for _ in range(1000)]
print(f"poorest-decile bias = {np.mean(bb):+.2f}  CI[{np.percentile(biasboot,2.5):+.2f}, {np.percentile(biasboot,97.5):+.2f}]")
# ---- 2. night-lights-ONLY baseline ----
M=pd.read_csv("data/processed/tile_cache_full/cache_metadata.csv")
cache=np.load("data/processed/tile_cache_full/cache.npy",mmap_mode="r")
si=rng.choice(len(M),9000,replace=False)
nl=np.array([float(np.asarray(cache[int(i)])[7].mean()) for i in si]); w=M.iloc[si].wealth_index_mean.values
b=np.polyfit(nl,w,1); pr=np.polyval(b,nl); r2nl=1-((w-pr)**2).sum()/((w-w.mean())**2).sum()
print("=== NIGHT-LIGHTS-ONLY baseline (1 feature) ===")
print(f"Spearman = {spearmanr(nl,w)[0]:.2f}   linear r2 = {r2nl:.2f}")
# ---- 3. pretrained baseline + spatial autocorr ----
print("=== pretrained / SAC results ===")
for pat in ["*pretrain*","*moran*","*sac*","*residual*"]:
    for f in glob.glob(f"results/{pat}.json")+glob.glob(f"results/**/{pat}.json",recursive=True):
        print(f"-- {f} --"); print(open(f).read()[:300])
