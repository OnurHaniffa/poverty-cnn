"""Train the 8-channel ResNet-18 on one or more cross-country folds.

Baseline run (from-scratch, default hyperparameters). For each fold:
  - train with Adam/MSE, ReduceLROnPlateau on val MSE, early-stop on val r^2
  - keep the best-val checkpoint, evaluate it on the held-out TEST countries
  - save checkpoint, per-cluster test predictions (+country), and a metrics JSON

Run on the PC GPU:
  PPY -m poverty_cnn.training.train --folds A --max-epochs 1        # smoke
  PPY -m poverty_cnn.training.train --folds A,B,C,D,E               # full overnight
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def r2_pearson(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    yt, pt = y - y.mean(), p - p.mean()
    pear = float((yt * pt).sum() / (np.sqrt((yt ** 2).sum() * (pt ** 2).sum()) + 1e-12))
    return r2, pear


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, ccs = [], [], []
    for x, y, meta in loader:
        ps.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        ccs.extend(meta["country"])
    return np.concatenate(ys), np.concatenate(ps), np.array(ccs)


def train_fold(fold, cache, out_dir, *, max_epochs, lr, patience, bs, workers,
               seed, device, channels=None, warmup=3, weight_decay=0.0,
               dropout=0.2, optimizer="adam"):
    set_seed(seed)
    loaders = make_fold_loaders(cache, fold, batch_size=bs, num_workers=workers,
                                channels=channels)
    in_ch = 8 if channels is None else len(channels)
    model = PovertyResNet(in_channels=in_ch, dropout=dropout).to(device)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_factor(ep):  # linear warmup -> cosine decay to 0 over max_epochs
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, max_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    lossf = nn.MSELoss()

    best_val, best_state, bad = -1e9, None, 0
    for ep in range(max_epochs):
        model.train()
        t0 = time.time()
        for x, y, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tame LR spikes
            opt.step()
        yv, pv, _ = evaluate(model, loaders["val"], device)
        val_mse = float(((yv - pv) ** 2).mean())
        val_r2, _ = r2_pearson(yv, pv)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"[{fold}] epoch {ep:3d}  val_r2 {val_r2:+.3f}  val_mse {val_mse:.3f}"
              f"  lr {lr_now:.1e}  ({time.time() - t0:.0f}s)", flush=True)
        if val_r2 > best_val:
            best_val = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{fold}] early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(best_state)
    yt, pt, cc = evaluate(model, loaders["test"], device)
    test_r2, test_pear = r2_pearson(yt, pt)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / f"model_fold{fold}.pt")
    np.savez(out / f"preds_fold{fold}.npz", y=yt, pred=pt, country=cc)
    metrics = {"fold": fold, "val_r2": round(best_val, 4),
               "test_r2": round(test_r2, 4), "test_pearson": round(test_pear, 4),
               "n_test": int(len(yt)), "epochs_run": ep + 1}
    json.dump(metrics, open(out / f"metrics_fold{fold}.json", "w"), indent=2)
    print(f"[{fold}] DONE  test_r2 {test_r2:+.3f}  pearson {test_pear:+.3f}  "
          f"n_test {len(yt)}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="A", help="comma-separated, e.g. A,B,C,D,E")
    ap.add_argument("--cache", default="data/processed/tile_cache")
    ap.add_argument("--out", default="results/cnn_baseline")
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)        # lowered from 1e-3 (stability)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)         # linear LR warmup epochs
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--channels", default="",
                    help="comma band indices, e.g. 0,1,2,3,4,5,6 (MS-only) or 7 (NL-only); empty = all 8")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True  # fixed 224x224 -> autotune conv algos

    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    results = []
    for fold in args.folds.split(","):
        fold = fold.strip()
        print(f"\n===== FOLD {fold} on {args.device} (channels={channels or 'all8'}) =====",
              flush=True)
        results.append(train_fold(
            fold, args.cache, args.out, max_epochs=args.max_epochs, lr=args.lr,
            patience=args.patience, bs=args.bs, workers=args.workers,
            seed=args.seed, device=args.device, channels=channels, warmup=args.warmup,
            weight_decay=args.weight_decay, dropout=args.dropout, optimizer=args.optimizer))

    if results:
        mean_r2 = float(np.mean([m["test_r2"] for m in results]))
        summary = {"folds": results, "mean_test_r2": round(mean_r2, 4),
                   "n_folds": len(results)}
        json.dump(summary, open(Path(args.out) / "summary.json", "w"), indent=2)
        print(f"\n===== MEAN test_r2 over {len(results)} folds: {mean_r2:+.3f} =====",
              flush=True)


if __name__ == "__main__":
    main()
