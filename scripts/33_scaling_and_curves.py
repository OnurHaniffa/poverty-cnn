"""Two evaluation upgrades (fold A, full cache):

A. DATA-SCALING CURVE — test r2 vs training-set fraction (20/40/60/80/100%). Visualises
   the central thesis ("the gap is data"): still climbing => more data helps; plateau =>
   ceiling reached.
B. LEARNING CURVE — train r2 AND val r2 per epoch (the classic overfitting graph): train
   keeps rising, val plateaus, early-stop fires.

Run on PC GPU0: PPY scripts/33_scaling_and_curves.py
"""
from __future__ import annotations
import math, random, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from poverty_cnn.data.dataset import PovertyTileDataset
from poverty_cnn.data import splits
from poverty_cnn.models.poverty_resnet import PovertyResNet

CACHE="data/processed/tile_cache_full"; DEV="cuda:0"; FOLD="A"
MAXEP=70; WARMUP=3; LR=3e-4; BS=64; SEED=42
NAVY="#1f3a5f"; TEAL="#178a7a"; AMBER="#e0922f"; GREY="#9aa4ae"
OUT=Path("results/figures/teaching"); OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"Avenir Next","axes.spines.top":False,"axes.spines.right":False})

def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)

@torch.no_grad()
def pr(model,loader):   # loader is an ALREADY-constructed DataLoader (don't re-wrap)
    ys,ps=[],[]
    for x,y,_ in loader:
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def main():
    cache=np.load(f"{CACHE}/cache.npy",mmap_mode="r"); meta=pd.read_csv(f"{CACHE}/cache_metadata.csv")
    stats=np.load(f"{CACHE}/norm_stats.npz"); mean,std=stats[f"{FOLD}_mean"],stats[f"{FOLD}_std"]
    tr_all=splits.clusters_for(meta,FOLD,"train"); va=splits.clusters_for(meta,FOLD,"val"); te=splits.clusters_for(meta,FOLD,"test")
    rng=np.random.default_rng(0)
    DS=lambda idx,aug: PovertyTileDataset(cache,meta,idx,mean,std,augment=aug)
    lva,lte=DataLoader(DS(va,False),256,num_workers=4),DataLoader(DS(te,False),256,num_workers=4)
    tr_sub=rng.choice(tr_all,2000,replace=False)  # for train-r2 monitoring

    def train(idx, log_curve=False):
        set_seed(SEED)
        ltr=DataLoader(DS(idx,True),BS,shuffle=True,num_workers=4,pin_memory=True)
        m=PovertyResNet(in_channels=8,dropout=0.2).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=LR)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/WARMUP if e<WARMUP else 0.5*(1+math.cos(math.pi*min((e-WARMUP)/max(1,MAXEP-WARMUP),1.0))))
        lf=nn.MSELoss(); best=-1e9; bst=None; bad=0; curve=[]
        for ep in range(MAXEP):
            m.train()
            for x,y,_ in ltr:
                x,y=x.to(DEV),y.to(DEV); opt.zero_grad(); l=lf(m(x),y); l.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            yv,pv=pr(m,lva); vr=r2(yv,pv); sch.step()
            if log_curve:
                yt,pt=pr(m,DataLoader(DS(tr_sub,False),256,num_workers=4)); curve.append((ep,float(r2(yt,pt)),float(vr)))
            if vr>best: best,bst,bad=vr,{k:v.cpu().clone() for k,v in m.state_dict().items()},0
            else:
                bad+=1
                if bad>=15: break
        m.load_state_dict(bst); yt,ptt=pr(m,lte); return float(r2(yt,ptt)), curve

    # ---- A: data-scaling ----
    fracs=[0.2,0.4,0.6,0.8,1.0]; pts=[]
    for f in fracs:
        n=int(len(tr_all)*f); idx=tr_all if f==1.0 else rng.choice(tr_all,n,replace=False)
        tr2,curve=train(idx, log_curve=(f==1.0))
        pts.append((n,tr2)); print(f"  frac {f:.0%} (n={n}): test r2 {tr2:+.3f}",flush=True)
        if f==1.0: full_curve=curve
    ns=[p[0] for p in pts]; rs=[p[1] for p in pts]
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(ns,rs,"-o",color=TEAL,lw=2.5,ms=8)
    ax.axhline(0.70,ls="--",color=GREY); ax.text(ns[0],0.71,"Yeh 2020 (0.70)",color=GREY,fontsize=10)
    ax.set_xlabel("training villages (fold A)"); ax.set_ylabel("held-out test r²")
    ax.set_title("Data-scaling: more data → higher r² (the gap is DATA)",color=NAVY,fontweight="bold")
    ax.grid(True,color="#eee"); fig.tight_layout(); fig.savefig(OUT/"data_scaling_curve.png",dpi=180,bbox_inches="tight",facecolor="white")
    json.dump({"n":ns,"test_r2":rs},open("results/data_scaling.json","w"))

    # ---- B: learning curve ----
    eps=[c[0] for c in full_curve]; trc=[c[1] for c in full_curve]; vac=[c[2] for c in full_curve]
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(eps,trc,"-o",color=AMBER,lw=2,ms=4,label="train r²")
    ax.plot(eps,vac,"-o",color=TEAL,lw=2,ms=4,label="validation r²")
    best_ep=int(np.argmax(vac)); ax.axvline(best_ep,ls="--",color=GREY); ax.text(best_ep+0.5,0.1,"early stop (best val)",color=GREY,fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("r²"); ax.legend(frameon=False)
    ax.set_title("Learning curve: train vs validation (overfitting check)",color=NAVY,fontweight="bold")
    ax.grid(True,color="#eee"); fig.tight_layout(); fig.savefig(OUT/"learning_curve.png",dpi=180,bbox_inches="tight",facecolor="white")
    print("\nwrote data_scaling_curve.png + learning_curve.png")
    print("data-scaling test r2:", [round(r,3) for r in rs])

if __name__=="__main__": main()
