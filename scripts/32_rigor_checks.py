"""Methodological-rigor evidence for the defense (overfitting + fuller metrics).

1. OVERFITTING DIAGNOSTIC: for the full model, evaluate each fold's checkpoint on its
   own TRAIN data (subsample) vs its held-out TEST data. A small train-test r2 gap =
   not overfit. (Cross-country CV already prevents memorisation; this SHOWS it.)
2. FULLER METRICS on the full-model held-out preds: RMSE (penalises big misses),
   plus precision/recall/F1 at the 'poorest-20%' poverty threshold (the classification
   framing for aid targeting).

Run on PC: PPY scripts/32_rigor_checks.py
"""
from __future__ import annotations
import json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from poverty_cnn.data.dataset import PovertyTileDataset
from poverty_cnn.data import splits
from poverty_cnn.models.poverty_resnet import PovertyResNet

CACHE="data/processed/tile_cache_full"; RUN="results/cnn_full"; DEV="cuda:0"; SUB=2500
def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)

@torch.no_grad()
def predict(model, ds):
    ys,ps=[],[]
    for x,y,_ in DataLoader(ds,batch_size=256,num_workers=4):
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def main():
    cache=np.load(f"{CACHE}/cache.npy",mmap_mode="r"); meta=pd.read_csv(f"{CACHE}/cache_metadata.csv")
    stats=np.load(f"{CACHE}/norm_stats.npz"); rng=np.random.default_rng(0)
    print("### 1. OVERFITTING DIAGNOSTIC (full model): train r2 vs held-out test r2")
    gaps=[]
    for fold in splits.fold_ids():
        mean,std=stats[f"{fold}_mean"],stats[f"{fold}_std"]
        model=PovertyResNet(in_channels=8,dropout=0.2).to(DEV)
        model.load_state_dict(torch.load(f"{RUN}/model_fold{fold}.pt",map_location=DEV)); model.eval()
        tr=splits.clusters_for(meta,fold,"train"); tr=rng.choice(tr,min(SUB,len(tr)),replace=False)
        te=splits.clusters_for(meta,fold,"test")
        ytr,ptr=predict(model,PovertyTileDataset(cache,meta,tr,mean,std,augment=False))
        yte,pte=predict(model,PovertyTileDataset(cache,meta,te,mean,std,augment=False))
        rt,re=r2(ytr,ptr),r2(yte,pte); gaps.append(rt-re)
        print(f"  fold {fold}: train r2 {rt:+.3f} | test r2 {re:+.3f} | gap {rt-re:+.3f}")
    print(f"  >>> mean train-test gap {np.mean(gaps):+.3f}  (small gap = NOT overfit; the model generalises)")

    # 2. fuller metrics on pooled held-out preds
    ys,ps=[],[]
    for fold in splits.fold_ids():
        z=np.load(f"{RUN}/preds_fold{fold}.npz"); ys.append(z["y"]);ps.append(z["pred"])
    y,p=np.concatenate(ys),np.concatenate(ps)
    rmse=np.sqrt(((y-p)**2).mean()); mae=np.abs(y-p).mean()
    thr=np.quantile(y,0.20); true_poor=y<=thr; pred_poor=p<=np.quantile(p,0.20)
    tp=(true_poor&pred_poor).sum(); prec=tp/max(1,pred_poor.sum()); rec=tp/max(1,true_poor.sum())
    f1=2*prec*rec/max(1e-9,prec+rec)
    print("\n### 2. FULLER METRICS (full model, pooled held-out)")
    print(f"  RMSE {rmse:.3f} | MAE {mae:.3f} | pooled r2 {r2(y,p):.3f}")
    print(f"  poorest-20% targeting: precision {prec:.1%} | recall {rec:.1%} | F1 {f1:.1%} (random ~20%)")

if __name__=="__main__": main()
