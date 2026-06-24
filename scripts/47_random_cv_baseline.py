"""Random-CV baseline — quantify the spatial-leakage inflation that our country-blocking AVOIDS.

Train ONE model on a RANDOM 80/20 cluster split (countries deliberately MIXED across train and
test) with the same config as cnn_full, and compare its held-out r2 to the country-blocked r2
(0.606). Random CV puts nearby / same-country clusters in BOTH train and test, so the test points
sit next to training points -> an inflated, leakage-prone estimate. The gap = the optimism our
leave-country-out design refuses to claim.

Run on PC GPU0 (detached): PPY scripts/47_random_cv_baseline.py
"""
from __future__ import annotations
import json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from poverty_cnn.data.dataset import PovertyTileDataset
from poverty_cnn.models.poverty_resnet import PovertyResNet

CACHE = "data/processed/tile_cache_full"; DEV = "cuda:0"
MAXEP = 70; WARMUP = 3; LR = 3e-4; BS = 64; SEED = 42


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def norm_stats(cache, idx):
    """Streaming per-channel mean/std over a sample of train tiles (avoid loading all)."""
    rng = np.random.default_rng(0)
    samp = rng.choice(idx, min(3000, len(idx)), replace=False)
    s = np.zeros(8); ss = np.zeros(8); cnt = 0
    for i in samp:
        x = np.asarray(cache[int(i)], dtype="float64")
        s += x.sum((1, 2)); ss += (x ** 2).sum((1, 2)); cnt += x.shape[1] * x.shape[2]
    mean = s / cnt; std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 1e-12))
    return mean.astype("float32"), std.astype("float32")


@torch.no_grad()
def ev(model, loader):
    model.eval(); ys, ps = [], []
    for x, y, _ in loader:
        ps.append(model(x.to(DEV)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    cache = np.load(f"{CACHE}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{CACHE}/cache_metadata.csv")
    n = len(meta); rng = np.random.default_rng(0); perm = rng.permutation(n)
    n_test = int(0.20 * n); te = perm[:n_test]; rest = perm[n_test:]
    n_val = int(0.10 * len(rest)); va = rest[:n_val]; tr = rest[n_val:]
    print(f"RANDOM split (countries mixed): train {len(tr)} | val {len(va)} | test {len(te)}", flush=True)
    mean, std = norm_stats(cache, tr)

    DS = lambda idx, aug: PovertyTileDataset(cache, meta, idx, mean, std, augment=aug)
    ltr = DataLoader(DS(tr, True), BS, shuffle=True, num_workers=4, pin_memory=True)
    lva = DataLoader(DS(va, False), 256, num_workers=4)
    lte = DataLoader(DS(te, False), 256, num_workers=4)

    model = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/WARMUP if e < WARMUP
                                            else 0.5*(1+math.cos(math.pi*min((e-WARMUP)/max(1, MAXEP-WARMUP), 1.0))))
    lossf = nn.MSELoss(); best = -1e9; bst = None; bad = 0
    for ep in range(MAXEP):
        model.train()
        for x, y, _ in ltr:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            loss = lossf(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        yv, pv = ev(model, lva); vr = r2(yv, pv); sch.step()
        if vr > best: best, bst, bad = vr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15: break
        if ep % 5 == 0: print(f"  ep{ep} val r2 {vr:+.3f}", flush=True)
    model.load_state_dict(bst)
    yt, pt = ev(model, lte)
    rr = float(r2(yt, pt)); sp = float(spearmanr(yt, pt)[0])
    print(f"\n=== RANDOM-CV held-out: r2 {rr:+.3f} | Spearman {sp:+.3f} ===")
    print(f"=== vs country-blocked (honest) r2 0.606 / Spearman 0.760 ===")
    print(f"=== inflation from random CV: r2 +{rr-0.606:.3f} ({(rr-0.606)/0.606*100:+.0f}%) ===")
    json.dump({"random_cv_r2": rr, "random_cv_spearman": sp,
               "country_blocked_r2": 0.606, "country_blocked_spearman": 0.760,
               "r2_inflation": rr - 0.606, "r2_inflation_pct": (rr - 0.606) / 0.606 * 100},
              open("results/random_cv_baseline.json", "w"), indent=2)
    print("wrote results/random_cv_baseline.json")


if __name__ == "__main__":
    main()
