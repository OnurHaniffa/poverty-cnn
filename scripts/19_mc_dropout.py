"""MC-dropout inference (contribution #3, the roadmap-named method).

Keep dropout ACTIVE at inference, run N stochastic forward passes per village.
mean = point estimate, std = uncertainty. CRITICAL: ResNet has BatchNorm — we
enable ONLY nn.Dropout (train mode) and keep BatchNorm in eval (running stats),
else BN would use batch stats and corrupt predictions.

Note: our model has a SINGLE dropout layer (before the head), so this is a weak
form of MC-dropout — we report it honestly alongside the deep ensemble.

Run on PC GPU: PPY scripts/19_mc_dropout.py
"""
from __future__ import annotations

import numpy as np
import torch

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

CACHE="data/processed/tile_cache"; RUN="results/cnn_stable"; N=30; DEV="cuda:0"


def enable_mc_dropout(model):
    model.eval()                                  # everything eval (BN uses running stats)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()                             # only dropout stays stochastic


def main():
    for fold in splits.fold_ids():
        model=PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
        model.load_state_dict(torch.load(f"{RUN}/model_fold{fold}.pt", map_location=DEV))
        enable_mc_dropout(model)
        loader=make_fold_loaders(CACHE, fold, batch_size=256, num_workers=4)["test"]
        means, stds, ys, ccs = [], [], [], []
        with torch.no_grad():
            for x, y, meta in loader:
                x=x.to(DEV)
                passes=torch.stack([model(x) for _ in range(N)])   # (N, b)
                means.append(passes.mean(0).cpu().numpy())
                stds.append(passes.std(0).cpu().numpy())
                ys.append(y.numpy()); ccs.extend(meta["country"])
        np.savez(f"{RUN}/mc_preds_fold{fold}.npz",
                 y=np.concatenate(ys), mc_mean=np.concatenate(means),
                 mc_std=np.concatenate(stds), country=np.array(ccs))
        print(f"  fold {fold}: mean mc_std {np.concatenate(stds).mean():.4f}")
    print("MC-dropout done ->", RUN, "/mc_preds_fold*.npz")


if __name__ == "__main__":
    main()
