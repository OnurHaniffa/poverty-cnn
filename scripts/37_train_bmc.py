"""Mitigation experiment #2: Balanced MSE (BMC) — a SECOND imbalanced-regression family.

Our LDS-style reweighting (scripts/26) did NOT rescue the poorest from regression-to-mean.
To defend "fundamental signal limit" rather than "we tried one trick", we test the dominant
post-LDS method: **Balanced MSE, BMC form** (Ren et al., CVPR 2022). BMC reframes regression
as a balanced softmax over the batch — logits = -(pred_i - y_j)^2 / (2*noise_var), diagonal
targets — which provably corrects the imbalanced-label prior WITHOUT hand-tuned bin weights.
noise_var is a LEARNED parameter (the homoscedastic noise scale), optimised jointly.

Same stabilized config + pilot cache + poorest-decile harness as scripts/26 -> clean A/B vs
cnn_stable (poorest-decile bias +0.66, poorest-20% recall 39%) and vs the LDS result.

Run on PC GPU0: PPY scripts/37_train_bmc.py --folds A,B,C,D,E
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet

CACHE="data/processed/tile_cache"; DEV="cuda:0"
MAXEP=80; WARMUP=3; LR=3e-4; BS=64; SEED=42


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)


def bmc_loss(pred, target, noise_sigma):
    """Balanced MSE (BMC). pred,target: (B,). noise_sigma: learnable scalar Parameter."""
    noise_var = noise_sigma ** 2
    logits = -(pred.unsqueeze(1) - target.unsqueeze(0)) ** 2 / (2 * noise_var)   # (B,B)
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss = F.cross_entropy(logits, labels)
    return loss * (2 * noise_var).detach()      # keep gradient scale ~ MSE


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps, cc = [], [], []
    for x, y, meta in loader:
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy()); cc.extend(meta["country"])
    return np.concatenate(ys), np.concatenate(ps), np.array(cc)


def poorest_metrics(y, p):
    dec0 = y <= np.quantile(y, 0.10)
    bias0 = float((p[dec0] - y[dec0]).mean())
    nt = int(len(y) * 0.20)
    rec20 = len(set(np.argsort(y)[:nt]) & set(np.argsort(p)[:nt])) / nt
    return bias0, rec20


def train_fold(fold, out_dir):
    set_seed(SEED)
    L = make_fold_loaders(CACHE, fold, batch_size=BS, num_workers=4)
    model = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
    noise_sigma = nn.Parameter(torch.tensor(1.0, device=DEV))    # learned BMC noise scale
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": LR},
                            {"params": [noise_sigma], "lr": 1e-2}])
    def lf(ep):
        if ep < WARMUP: return (ep+1)/WARMUP
        return 0.5*(1+math.cos(math.pi*min((ep-WARMUP)/max(1,MAXEP-WARMUP),1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lf, lf])

    best=-1e9; best_state=None; bad=0
    for ep in range(MAXEP):
        model.train()
        for x,y,_ in L["train"]:
            x=x.to(DEV); yd=y.to(DEV)
            opt.zero_grad()
            loss=bmc_loss(model(x), yd, noise_sigma)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        yv,pv,_=evaluate(model,L["val"]); vr=r2(yv,pv); sched.step()
        if vr>best: best,best_state,bad=vr,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else:
            bad+=1
            if bad>=15: break
    model.load_state_dict(best_state)
    y,p,cc=evaluate(model,L["test"])
    np.savez(f"{out_dir}/preds_fold{fold}.npz", y=y, pred=p, country=cc)
    b0,r20=poorest_metrics(y,p)
    m={"fold":fold,"test_r2":round(float(r2(y,p)),4),"poorest_bias":round(b0,4),
       "poorest20_recall":round(r20,4),"noise_sigma":round(float(noise_sigma.item()),4)}
    print(f"[{fold}] r2 {m['test_r2']:+.3f} | poorest-decile bias {b0:+.3f} | "
          f"poorest-20% recall {r20:.1%} | learned noise_sigma {m['noise_sigma']}", flush=True)
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds", default="A,B,C,D,E")
    ap.add_argument("--out", default="results/cnn_bmc")
    args=ap.parse_args()
    torch.backends.cudnn.benchmark=True
    Path(args.out).mkdir(parents=True, exist_ok=True)
    res=[train_fold(f.strip(), args.out) for f in args.folds.split(",")]
    json.dump({"folds":res,
               "mean_test_r2":round(float(np.mean([m["test_r2"] for m in res])),4),
               "mean_poorest_bias":round(float(np.mean([m["poorest_bias"] for m in res])),4),
               "mean_poorest20_recall":round(float(np.mean([m["poorest20_recall"] for m in res])),4)},
              open(f"{args.out}/summary.json","w"), indent=2)
    print(f"\n=== BMC vs baseline cnn_stable (poorest bias +0.66, recall 39%) and vs LDS (no fix) ===")


if __name__ == "__main__":
    main()
