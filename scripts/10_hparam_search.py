"""Optuna hyperparameter search for the 8-channel ResNet-18 (fold A, train/val only).

Searches lr, weight_decay, dropout, optimizer (adam vs sgd+momentum), warmup.
Trains on fold A's TRAIN split, scores on fold A's VAL split (TEST never touched
-> no leakage). TPE sampler + median pruner (kills hopeless trials early).
Writes the winner to <out>/best_params.json for the final tuned 5-fold.

Run on the PC, GPU 0 only:
  PPY scripts/10_hparam_search.py --trials 25 --max-epochs 40 --out results/optuna
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


@torch.no_grad()
def val_r2(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, y, _ in loader:
        ps.append(model(x.to(device)).cpu().numpy()); ys.append(y.numpy())
    return float(r2(np.concatenate(ys), np.concatenate(ps)))


def make_objective(cache, device, max_epochs, patience, bs, workers):
    def objective(trial):
        set_seed(42)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        opt_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        warmup = trial.suggest_int("warmup", 2, 5)

        L = make_fold_loaders(cache, "A", batch_size=bs, num_workers=workers)
        model = PovertyResNet(in_channels=8, dropout=dropout).to(device)
        if opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        def lf(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            prog = (ep - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
        lossf = nn.MSELoss()

        best, bad = -1e9, 0
        for ep in range(max_epochs):
            model.train()
            for x, y, _ in L["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = lossf(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            vr2 = val_r2(model, L["val"], device)
            sched.step()
            trial.report(vr2, ep)
            if vr2 > best:
                best, bad = vr2, 0
            else:
                bad += 1
                if bad >= patience:
                    break
            if trial.should_prune():
                raise optuna.TrialPruned()
        print(f"  trial {trial.number}: best_val_r2={best:.4f}  {trial.params}", flush=True)
        return best
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed/tile_cache")
    ap.add_argument("--out", default="results/optuna")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    Path(args.out).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))
    obj = make_objective(args.cache, args.device, args.max_epochs, args.patience,
                         args.bs, args.workers)
    t0 = time.time()
    study.optimize(obj, n_trials=args.trials)

    print("\n=== best val_r2: %.4f ===" % study.best_value, flush=True)
    print("best params:", study.best_params, flush=True)
    json.dump({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "all_trials": [{"value": t.value, "params": t.params, "state": str(t.state)}
                       for t in study.trials],
    }, open(Path(args.out) / "best_params.json", "w"), indent=2)
    print("wrote %s in %.0f min" % (Path(args.out) / "best_params.json",
                                    (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
