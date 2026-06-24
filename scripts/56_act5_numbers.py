import numpy as np, pandas as pd, glob
from scipy.stats import spearmanr
print("=== fairness_audit_percountry.csv ===")
df=pd.read_csv("results/fairness_audit_percountry.csv")
print("cols:", list(df.columns))
print(df.head(3).to_string())
# urban vs rural spearman columns?
cu=[c for c in df.columns if 'urban' in c.lower() and 'spear' in c.lower()]
cr=[c for c in df.columns if 'rural' in c.lower() and 'spear' in c.lower()]
print("urban col:",cu," rural col:",cr)
if cu and cr:
    u=df[cu[0]]; r=df[cr[0]]
    print(f"median urban spearman={u.median():.3f}  rural={r.median():.3f}  rural<urban in {int((r<u).sum())}/{len(df)} countries")

print("=== regression-to-mean from predictions (cnn_stable) ===")
Y=[];P=[]
for f in sorted(glob.glob("results/cnn_stable/preds_fold*.npz")):
    z=np.load(f,allow_pickle=True); Y+=list(z["y"]);P+=list(z["pred"])
Y=np.array(Y,float);P=np.array(P,float)
slope,intercept=np.polyfit(Y,P,1)
print(f"slope of pred~true = {slope:.3f} (1.0=perfect; <1 = regression to mean)  n={len(Y)}")
# bias by decile of true wealth
dec=pd.qcut(Y,10,labels=False)
print("decile | mean_true | mean_bias(pred-true)")
biases=[]
for d in range(10):
    m=dec==d; b=(P[m]-Y[m]).mean(); biases.append(b)
    print(f"  {d}: true={Y[m].mean():+.2f}  bias={b:+.2f}")
print("poorest decile over-predicted by", round(float(biases[0]),2), "| richest under by", round(float(biases[-1]),2))
