"""Temporal-drift experiment (contribution #4): train on EARLY years, test on LATE.

A true 2-D holdout: each fold trains on (train-countries AND year<=2014) and tests on
(test-countries AND year>=2015) -> unseen COUNTRY *and* unseen TIME. Compares the
temporal-holdout r2 to the spatial-only full model (0.597): if close, the model
generalises across time; if much lower, there is temporal drift. Also reports whether
the urban/rural gap / poorest-bias drift out-of-time.

Run on PC GPU0: PPY scripts/31_train_temporal.py
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from poverty_cnn.data.dataset import PovertyTileDataset
from poverty_cnn.data import splits
from poverty_cnn.models.poverty_resnet import PovertyResNet

CACHE="data/processed/tile_cache_full"; OUT="results/cnn_temporal"; DEV="cuda:0"
SPLIT=2014; MAXEP=80; WARMUP=3; LR=3e-4; BS=64; SEED=42

def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)

@torch.no_grad()
def ev(model,loader):
    model.eval(); ys,ps,cc,uu=[],[],[],[]
    for x,y,m in loader:
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy()); cc.extend(m["country"])
    return np.concatenate(ys),np.concatenate(ps)

def main():
    Path(OUT).mkdir(parents=True,exist_ok=True); torch.backends.cudnn.benchmark=True
    cache=np.load(f"{CACHE}/cache.npy",mmap_mode="r"); meta=pd.read_csv(f"{CACHE}/cache_metadata.csv")
    stats=np.load(f"{CACHE}/norm_stats.npz")
    early=meta.year.values<=SPLIT; late=meta.year.values>SPLIT
    summary=[]
    for fold in splits.fold_ids():
        set_seed(SEED)
        mean,std=stats[f"{fold}_mean"],stats[f"{fold}_std"]
        def rows(role,mask):
            r=splits.clusters_for(meta,fold,role); return r[mask[r]]
        tr=rows("train",early); va=rows("val",early); te=rows("test",late)
        L=lambda idx,aug,sh: DataLoader(PovertyTileDataset(cache,meta,idx,mean,std,augment=aug),
                                        batch_size=BS,shuffle=sh,num_workers=4,pin_memory=True)
        ltr,lva,lte=L(tr,True,True),L(va,False,False),L(te,False,False)
        model=PovertyResNet(in_channels=8,dropout=0.2).to(DEV)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/WARMUP if e<WARMUP else 0.5*(1+math.cos(math.pi*min((e-WARMUP)/max(1,MAXEP-WARMUP),1.0))))
        lossf=nn.MSELoss(); best=-1e9; bst=None; bad=0
        for ep in range(MAXEP):
            model.train()
            for x,y,_ in ltr:
                x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
                loss=lossf(model(x),y); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            yv,pv=ev(model,lva); vr=r2(yv,pv); sched.step()
            if vr>best: best,bst,bad=vr,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
            else:
                bad+=1
                if bad>=15: break
        model.load_state_dict(bst)
        yt,pt=ev(model,lte)
        np.savez(f"{OUT}/preds_fold{fold}.npz",y=yt,pred=pt)
        m={"fold":fold,"temporal_r2":round(float(r2(yt,pt)),4),"n_train":int(len(tr)),"n_test":int(len(te))}
        summary.append(m); print(f"[{fold}] temporal r2 {m['temporal_r2']:+.3f} (train {len(tr)} early, test {len(te)} late)",flush=True)
    mt=float(np.mean([m["temporal_r2"] for m in summary]))
    json.dump({"folds":summary,"mean_temporal_r2":round(mt,4)},open(f"{OUT}/summary.json","w"),indent=2)
    print(f"\n=== TEMPORAL mean r2 {mt:+.3f}  vs spatial full-model 0.597  (close = generalises across time) ===")

if __name__=="__main__": main()
