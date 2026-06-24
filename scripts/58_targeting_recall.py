import numpy as np, glob
Y=[];P=[]
for f in sorted(glob.glob("results/cnn_stable/preds_fold*.npz")):
    z=np.load(f,allow_pickle=True); Y+=list(z['y']);P+=list(z['pred'])
Y=np.array(Y,float);P=np.array(P,float);n=len(Y)
for q in [0.10,0.20,0.25]:
    k=int(n*q)
    truly=set(np.argsort(Y)[:k])          # truly poorest q
    targeted=set(np.argsort(P)[:k])        # who the model would target
    recall=len(truly&targeted)/k
    print(f"target poorest {int(q*100)}%: recall = {recall:.2f}  ->  misses {1-recall:.0%} of the truly-neediest")
