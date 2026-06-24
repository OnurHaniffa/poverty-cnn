import json, numpy as np, glob
from scipy.stats import pearsonr
print("=== UNCERTAINTY: does it flag the poorest? ===")
u=json.load(open("results/uncertainty_proper.json"))
print(f"{'method':16} {'AURG':>7} {'cov_poor20':>11} {'cov_rich20':>11}")
for m,v in u.items():
    if isinstance(v,dict) and 'aurg' in v:
        print(f"{m:16} {v['aurg']:>7.3f} {v.get('cov_poorest20',0):>11.3f} {v.get('cov_richest20',0):>11.3f}")
print("=== MITIGATION: does reweighting fix the poorest-bias? ===")
def metrics(d):
    Y=[];P=[]
    for f in sorted(glob.glob(f"{d}/preds_fold*.npz")):
        z=np.load(f,allow_pickle=True); Y+=list(z['y']);P+=list(z['pred'])
    if not Y: return None
    Y=np.array(Y,float);P=np.array(P,float)
    r2=1-((Y-P)**2).sum()/((Y-Y.mean())**2).sum()
    import pandas as pd
    dec=pd.qcut(Y,10,labels=False)
    poorbias=(P[dec==0]-Y[dec==0]).mean()
    return r2,poorbias,len(Y)
for d in ["results/cnn_stable","results/cnn_reweighted","results/cnn_reweighted_a10","results/cnn_reweighted_a15","results/cnn_reweighted_asym"]:
    r=metrics(d)
    if r: print(f"  {d.split('/')[-1]:22} r2={r[0]:+.3f}  poorest_decile_bias={r[1]:+.3f}  n={r[2]}")
