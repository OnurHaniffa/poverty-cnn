import numpy as np, glob, json, os
from scipy.stats import pearsonr, spearmanr
print("=== leave-country-out pooled + worst-group ===")
for d in ["results/cnn_tuned","results/cnn_stable_s3","results/cnn_stable","results/cnn_baseline"]:
    fs=sorted(glob.glob(f"{d}/preds_fold*.npz"))
    if not fs: continue
    Y=[];P=[];C=[]
    for f in fs:
        z=np.load(f,allow_pickle=True); Y+=list(z["y"]);P+=list(z["pred"]);C+=list(z["country"])
    Y=np.array(Y,float);P=np.array(P,float);C=np.array([str(x) for x in C])
    r=pearsonr(Y,P)[0]; r2=1-((Y-P)**2).sum()/((Y-Y.mean())**2).sum()
    pc={c:round(float(pearsonr(Y[C==c],P[C==c])[0]),3) for c in set(C) if (C==c).sum()>=30}
    print(f"[{os.path.basename(d)}] n={len(Y)} pooled_r={r:.3f} r2={r2:.3f} worst_group_r={min(pc.values()):.3f} n_countries={len(pc)}")

print("=== channel ablation files ===")
for f in sorted(glob.glob("results/**/*ablat*",recursive=True)+glob.glob("results/**/*channel*",recursive=True)): print(" ",f)
print("=== OOD per-country spearman ===")
rr=json.load(open("results/ood_rigor.json"))
for k,v in rr.items():
    print("  %s: spearman=%.3f r2=%.3f beats_baseline=%s"%(k,v["spearman"],v.get("r2",0),v.get("beats_baseline")))
