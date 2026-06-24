"""OOD rigor pass: bootstrap 95% CIs (per country) + trivial baseline + targeting recall.

Re-runs the frozen 36k ensemble (fast) and adds the statistics the OOD test was missing:
- cluster-level bootstrap 95% CIs on per-country Spearman & r2 (is South Africa SIGNIFICANTLY
  worse, or just small-n noise? e.g. Niger n=207)
- a trivial "predict the country mean" baseline (MAE_base): if the model's MAE beats it, the OOD
  signal is real; if not (negative r2), the model is worse than guessing — made concrete
- targeting recall @ poorest-20% (the policy metric that complements Spearman)

Run on PC GPU0: PPY scripts/45_ood_rigor.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

OOD = "data/processed/tile_cache_ood"; DEV = "cuda:0"; BS = 256
RUN = "results/cnn_full"; NORM = "data/processed/tile_cache_full/norm_stats.npz"
NAMES = {"ZA": "S.Africa", "NM": "Namibia", "GA": "Gabon", "SZ": "Eswatini", "MD": "Madagascar", "NI": "Niger"}


def r2(y, p): return 1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


@torch.no_grad()
def ensemble_predict(cache):
    stats = np.load(NORM); fps = []
    for fold in splits.fold_ids():
        m_ = stats[f"{fold}_mean"].reshape(1, 8, 1, 1); s_ = stats[f"{fold}_std"].reshape(1, 8, 1, 1)
        net = PovertyResNet(in_channels=8, dropout=0.2).to(DEV)
        net.load_state_dict(torch.load(f"{RUN}/model_fold{fold}.pt", map_location=DEV)); net.eval()
        pr = []
        for i0 in range(0, len(cache), BS):
            x = (np.asarray(cache[i0:i0+BS], dtype="float32") - m_) / s_
            pr.append(net(torch.from_numpy(x).to(DEV)).cpu().numpy())
        fps.append(np.concatenate(pr))
    return np.mean(fps, axis=0)


def boot_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(y)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, N, N) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def recall_at(y, p, q=0.2):
    nt = max(1, int(len(y) * q))
    return len(set(np.argsort(y)[:nt]) & set(np.argsort(p)[:nt])) / nt


def main():
    cache = np.load(f"{OOD}/cache.npy", mmap_mode="r")
    meta = pd.read_csv(f"{OOD}/cache_metadata.csv")
    y = meta.wealth_index_mean.values.astype("float32"); cc = meta.country.values
    pred = ensemble_predict(cache)
    out = {}
    print("=== 36k OOD: per-country bootstrap 95% CI + baseline + targeting ===")
    for c in sorted(np.unique(cc)):
        m = cc == c; yc, pc = y[m], pred[m]
        sp = float(spearmanr(yc, pc)[0]); sp_lo, sp_hi = boot_ci(yc, pc, lambda a, b: spearmanr(a, b)[0])
        r2v = float(r2(yc, pc)); r2_lo, r2_hi = boot_ci(yc, pc, lambda a, b: r2(a, b))
        mae = float(np.abs(yc - pc).mean()); mae_base = float(np.abs(yc - yc.mean()).mean())
        rec = float(recall_at(yc, pc))
        out[c] = dict(spearman=sp, spearman_ci=[sp_lo, sp_hi], r2=r2v, r2_ci=[r2_lo, r2_hi],
                      mae=mae, mae_baseline=mae_base, beats_baseline=mae < mae_base,
                      recall_poorest20=rec, n=int(m.sum()))
        flag = "✓beats-mean" if mae < mae_base else "✗WORSE-than-mean"
        print(f"  {NAMES.get(c,c):11} Spearman {sp:+.3f} [{sp_lo:+.3f},{sp_hi:+.3f}] | "
              f"r2 {r2v:+.3f} [{r2_lo:+.3f},{r2_hi:+.3f}] | MAE {mae:.3f} vs mean {mae_base:.3f} {flag} | "
              f"recall@20% {rec:.0%} | n={int(m.sum())}")
    # pooled
    sp = float(spearmanr(y, pred)[0]); lo, hi = boot_ci(y, pred, lambda a, b: spearmanr(a, b)[0])
    print(f"\n  POOLED Spearman {sp:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] | overall recall@20% {recall_at(y,pred):.0%}")
    json.dump(out, open("results/ood_rigor.json", "w"), indent=2)
    print("wrote results/ood_rigor.json")


if __name__ == "__main__":
    main()
