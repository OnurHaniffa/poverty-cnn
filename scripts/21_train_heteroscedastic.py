"""Heteroscedastic (aleatoric) uncertainty — the principled, stable version.

End-to-end NLL destabilises the mean (variance head 'explains away' errors). So
instead: take the ALREADY-VALIDATED MSE model as a FROZEN mean predictor, and
train ONLY a small variance head on its frozen 512-d features, with Gaussian NLL.

This keeps the mean exactly = the replication model (r2 ~0.56, no degradation) and
asks the sharp question: do the model's learned features carry information about
WHERE it errs (aleatoric uncertainty)? corr(predicted sigma, |error|) answers it.

Run on PC GPU: PPY scripts/21_train_heteroscedastic.py
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from poverty_cnn.data.dataset import make_fold_loaders
from poverty_cnn.models.poverty_resnet import PovertyResNet
from poverty_cnn.data import splits

CACHE="data/processed/tile_cache"; BASE="results/cnn_stable"; OUTDIR="results/cnn_hetero"
DEV="cuda:0"; EPOCHS=40; LR=1e-3


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def r2(y,p): return 1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12)


class VarHead(nn.Module):
    """Frozen trained backbone+mean head; trainable variance head on the features."""
    def __init__(self, base: PovertyResNet):
        super().__init__()
        self.base=base.eval()
        for p in self.base.parameters(): p.requires_grad_(False)
        self.var=nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Linear(128,1))

    def feats(self, x):
        n=self.base.net
        x=n.conv1(x); x=n.bn1(x); x=n.relu(x); x=n.maxpool(x)
        x=n.layer1(x); x=n.layer2(x); x=n.layer3(x); x=n.layer4(x)
        return torch.flatten(n.avgpool(x),1)            # (B,512)

    def forward(self, x):
        with torch.no_grad():
            f=self.feats(x); mu=self.base.net.fc(f).squeeze(-1)   # frozen mean
        return mu, self.var(f).squeeze(-1)              # mu (fixed), log_var (trained)


def nll(mu, log_var, y):
    log_var=log_var.clamp(-6,4)
    return 0.5*(log_var + (y-mu)**2 / log_var.exp()).mean()


@torch.no_grad()
def evaluate(model, loader):
    model.var.eval(); ys,mus,sg,cc=[],[],[],[]
    for x,y,meta in loader:
        mu,lv=model(x.to(DEV)); lv=lv.clamp(-6,4)
        ys.append(y.numpy()); mus.append(mu.cpu().numpy()); sg.append(np.exp(0.5*lv.cpu().numpy())); cc.extend(meta["country"])
    return np.concatenate(ys),np.concatenate(mus),np.concatenate(sg),np.array(cc)


def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark=True
    summary=[]
    for fold in splits.fold_ids():
        set_seed(42)
        base=PovertyResNet(in_channels=8,dropout=0.2).to(DEV)
        base.load_state_dict(torch.load(f"{BASE}/model_fold{fold}.pt",map_location=DEV))
        model=VarHead(base).to(DEV)
        opt=torch.optim.Adam(model.var.parameters(), lr=LR)
        L=make_fold_loaders(CACHE, fold, batch_size=128, num_workers=4)
        best=1e9; best_state=None; bad=0
        for ep in range(EPOCHS):
            model.var.train()
            for x,y,_ in L["train"]:
                x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
                mu,lv=model(x); loss=nll(mu,lv,y); loss.backward(); opt.step()
            yv,muv,sgv,_=evaluate(model,L["val"])
            vnll=float(0.5*(np.log(sgv**2+1e-9)+(yv-muv)**2/(sgv**2+1e-9)).mean())  # val NLL (select on THIS)
            if vnll<best: best,best_state,bad=vnll,{k:v.cpu().clone() for k,v in model.var.state_dict().items()},0
            else:
                bad+=1
                if bad>=8: break
        model.var.load_state_dict(best_state)
        y,mu,sigma,cc=evaluate(model,L["test"])
        c=float(np.corrcoef(sigma,np.abs(y-mu))[0,1])
        np.savez(f"{OUTDIR}/preds_fold{fold}.npz", y=y, mean=mu, sigma=sigma, country=cc)
        m={"fold":fold,"test_r2":round(float(r2(y,mu)),4),"corr_sigma_err":round(c,4),"mean_sigma":round(float(sigma.mean()),4)}
        summary.append(m); print(f"[{fold}] r2 {m['test_r2']:+.3f} (frozen mean)  corr(sigma,err) {c:+.3f}", flush=True)
    mc=float(np.mean([m["corr_sigma_err"] for m in summary]))
    json.dump({"folds":summary,"mean_test_r2":round(float(np.mean([m['test_r2'] for m in summary])),4),
               "mean_corr_sigma_err":round(mc,4)}, open(f"{OUTDIR}/summary.json","w"), indent=2)
    print(f"\n=== heteroscedastic: mean corr(sigma,err) {mc:+.3f}  (>>0 = aleatoric uncertainty WORKS) ===")


if __name__ == "__main__":
    main()
