"""Pretrained-vs-from-scratch ablation (fold A) — settle the 'gap to Yeh' attribution.

Train an ImageNet-pretrained ResNet-18 (8-channel stem: pretrained RGB weights copied into
the first 3 channels, the other 5 initialised from the RGB mean) on fold A's country-blocked
train set, same stabilised config, and compare its held-out r² to the from-scratch fold A
(0.690). The delta = the from-scratch penalty we currently only assert.

Run on PC GPU0 (detached): PPY scripts/52_pretrained_baseline.py
"""
from __future__ import annotations
import json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from scipy.stats import spearmanr
from poverty_cnn.data.dataset import PovertyTileDataset
from poverty_cnn.data import splits

CACHE = "data/processed/tile_cache_full"; DEV = "cuda:0"; FOLD = "A"
MAXEP = 60; WARMUP = 3; LR = 3e-4; BS = 64; SEED = 42; FROM_SCRATCH_R2 = 0.690


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def build_pretrained():
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    old = net.conv1.weight.data.clone()                 # (64,3,7,7)
    net.conv1 = nn.Conv2d(8, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        net.conv1.weight[:, :3] = old
        net.conv1.weight[:, 3:] = old.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
    return net


@torch.no_grad()
def ev(model, loader):
    model.eval(); ys, ps = [], []
    for x, y, _ in loader:
        ps.append(model(x.to(DEV)).squeeze(-1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    set_seed(SEED); torch.backends.cudnn.benchmark = True
    cache = np.load(f"{CACHE}/cache.npy", mmap_mode="r"); meta = pd.read_csv(f"{CACHE}/cache_metadata.csv")
    stats = np.load(f"{CACHE}/norm_stats.npz"); mean, std = stats[f"{FOLD}_mean"], stats[f"{FOLD}_std"]
    tr = splits.clusters_for(meta, FOLD, "train"); va = splits.clusters_for(meta, FOLD, "val"); te = splits.clusters_for(meta, FOLD, "test")
    DS = lambda idx, aug: PovertyTileDataset(cache, meta, idx, mean, std, augment=aug)
    ltr = DataLoader(DS(tr, True), BS, shuffle=True, num_workers=4, pin_memory=True)
    lva = DataLoader(DS(va, False), 256, num_workers=4); lte = DataLoader(DS(te, False), 256, num_workers=4)
    print(f"fold {FOLD}: train {len(tr)} val {len(va)} test {len(te)} | ImageNet-pretrained stem", flush=True)

    model = build_pretrained().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/WARMUP if e < WARMUP
                                            else 0.5*(1+math.cos(math.pi*min((e-WARMUP)/max(1, MAXEP-WARMUP), 1.0))))
    lossf = nn.MSELoss(); best = -1e9; bst = None; bad = 0
    for ep in range(MAXEP):
        model.train()
        for x, y, _ in ltr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            loss = lossf(model(x).squeeze(-1), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv, pv = ev(model, lva); vr = r2(yv, pv); sch.step()
        if vr > best: best, bst, bad = vr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
        if ep % 5 == 0: print(f"  ep{ep} val r2 {vr:+.3f}", flush=True)
    model.load_state_dict(bst)
    yt, pt = ev(model, lte); rr = float(r2(yt, pt)); sp = float(spearmanr(yt, pt)[0])
    print(f"\n=== PRETRAINED fold A: r2 {rr:+.3f} | Spearman {sp:+.3f} ===")
    print(f"=== from-scratch fold A: r2 {FROM_SCRATCH_R2:+.3f}  → pretraining delta {rr-FROM_SCRATCH_R2:+.3f} ===")
    json.dump({"pretrained_r2": rr, "pretrained_spearman": sp, "from_scratch_r2": FROM_SCRATCH_R2,
               "pretraining_gain": rr - FROM_SCRATCH_R2}, open("results/pretrained_baseline.json", "w"), indent=2)
    print("wrote results/pretrained_baseline.json")


if __name__ == "__main__":
    main()
