"""Mitigation experiment: can a reweighted (imbalanced-regression) loss rescue the
poorest villages from the model's regression-to-the-mean?

Our audit found the model is CONFIDENTLY WRONG about the poorest (slope 0.60, poorest
decile predicted +0.66 too rich, largest error there). That is a textbook deep
imbalanced-regression failure (Yang 2021, ICML). Mitigation = Label-Distribution-
Smoothing (LDS)-style sample reweighting: weight each village by (1/smoothed label
density)^alpha, so the sparse wealth TAILS (poorest + richest) get upweighted and the
model stops collapsing toward the dense middle.

Same stabilized config as cnn_stable, ONLY the loss weighting differs -> clean A/B vs
results/cnn_stable. Saves preds (y, pred, country) so the SAME audit (calibration,
poorest-decile bias, targeting recall) can be re-run and compared.

Run on PC GPU0: PPY scripts/26_train_reweighted.py --folds A           # quick prototype
                PPY scripts/26_train_reweighted.py --folds A,B,C,D,E   # full
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

CACHE="data/processed/tile_cache"; DEV="cuda:0"
MAXEP=80; WARMUP=3; LR=3e-4; BS=64; SEED=42


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)


def lds_weights(labels, n_bins=50, sigma=2.0, alpha=0.5, wmax=5.0, asymmetric=False):
    """Return (edges, per-bin weights) for LDS-style inverse-density reweighting.

    asymmetric=True: only upweight the POOR tail (below median wealth); leave the
    rich at weight 1 — targets the poorest without wasting weight on the rich tail.
    """
    hist, edges = np.histogram(labels, bins=n_bins)
    smooth = np.maximum(gaussian_filter1d(hist.astype(float), sigma), 1.0)
    w = (1.0 / smooth) ** alpha
    if asymmetric:
        centers = 0.5 * (edges[:-1] + edges[1:])
        w = np.where(centers <= np.median(labels), w, 1.0)   # poor only
    # normalize so the mean weight over the DATA is ~1 (keeps loss scale comparable)
    w = w / np.average(w, weights=np.maximum(hist, 1e-9))
    return edges, np.minimum(w, wmax)


def weight_of(y, edges, binw):
    idx = np.clip(np.digitize(y, edges) - 1, 0, len(binw) - 1)
    return binw[idx].astype("float32")


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps, cc = [], [], []
    for x, y, meta in loader:
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy()); cc.extend(meta["country"])
    return np.concatenate(ys), np.concatenate(ps), np.array(cc)


def poorest_metrics(y, p):
    """Bias and recall on the poorest decile/quintile — what the mitigation targets."""
    dec0 = y <= np.quantile(y, 0.10)
    bias0 = float((p[dec0] - y[dec0]).mean())     # >0 = predicted too rich
    nt = int(len(y) * 0.20)
    rec20 = len(set(np.argsort(y)[:nt]) & set(np.argsort(p)[:nt])) / nt
    return bias0, rec20


def train_fold(fold, alpha, out_dir, asymmetric=False):
    set_seed(SEED)
    L = make_fold_loaders(CACHE, fold, batch_size=BS, num_workers=4)
    train_labels = L["train"].dataset.wealth[L["train"].dataset.rows]
    edges, binw = lds_weights(train_labels, alpha=alpha, asymmetric=asymmetric)

    model = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    def lf(ep):
        if ep < WARMUP: return (ep+1)/WARMUP
        return 0.5*(1+math.cos(math.pi*min((ep-WARMUP)/max(1,MAXEP-WARMUP),1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)

    best=-1e9; best_state=None; bad=0
    for ep in range(MAXEP):
        model.train()
        for x,y,_ in L["train"]:
            x=x.to(DEV); yd=y.to(DEV)
            w=torch.from_numpy(weight_of(y.numpy(), edges, binw)).to(DEV)
            opt.zero_grad()
            loss=(w*(model(x)-yd)**2).mean()          # weighted MSE
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
    m={"fold":fold,"test_r2":round(float(r2(y,p)),4),"poorest_bias":round(b0,4),"poorest20_recall":round(r20,4)}
    print(f"[{fold}] r2 {m['test_r2']:+.3f} | poorest-decile bias {b0:+.3f} | poorest-20% recall {r20:.1%}", flush=True)
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds", default="A")
    ap.add_argument("--alpha", type=float, default=0.5, help="reweight strength (0=baseline,1=full inverse density)")
    ap.add_argument("--asymmetric", action="store_true", help="upweight only the POOR tail")
    ap.add_argument("--out", default="results/cnn_reweighted")
    args=ap.parse_args()
    torch.backends.cudnn.benchmark=True
    Path(args.out).mkdir(parents=True, exist_ok=True)
    res=[m for f in args.folds.split(",") for m in [train_fold(f.strip(), args.alpha, args.out, args.asymmetric)]]
    json.dump({"folds":res,"alpha":args.alpha,
               "mean_test_r2":round(float(np.mean([m["test_r2"] for m in res])),4)},
              open(f"{args.out}/summary.json","w"), indent=2)
    print(f"\n=== reweighted (alpha={args.alpha}): vs cnn_stable poorest-decile bias +0.66, poorest-20% recall 39% ===")


if __name__ == "__main__":
    main()
