"""OOD wealth labels via the FROZEN training PCA (no leakage).

The 23-country model predicts wealth on the pooled-PCA axis. To test it on NEW countries
we must put their villages on the SAME ruler — i.e. PROJECT their household assets through
the PCA fit on the 23 training countries, never refit. This script:

  1. Refits the pooled PCA on the 23-country training HR and CAPTURES the exact frozen
     transform (feature means/stds, PC1 loadings, PC1 centering/scaling, sign convention).
  2. VALIDATES the refit reproduces data/processed/multiround_wealth_index_clusters.csv
     (the labels the model was actually trained on) — correlation must be ~1.0.
  3. Projects each OOD country's households through the frozen transform -> household wealth
     -> cluster wealth, on the identical axis. Imputation for OOD uses the within-survey mean
     then falls back to the TRAINING global mean (never OOD distribution stats), keeping it frozen.

Output: data/processed/ood_wealth_index_clusters.csv
  columns: country, year, cluster_id, wealth_index_mean, n_households

Run on PC: PPY scripts/38_ood_wealth_project.py
"""
from __future__ import annotations
import tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from poverty_cnn.data.dhs import extract_asset_features, load_dhs_hr

TRAIN_MANIFEST = "data/raw/dhs_multiround_manifest.csv"; TRAIN_RAW = Path("data/raw/dhs_multiround")
OOD_MANIFEST = "data/raw/dhs_ood_manifest.csv"; OOD_RAW = Path("data/raw/dhs_ood")
TRAIN_LABELS = "data/processed/multiround_wealth_index_clusters.csv"
ERA = (2008, 2022); OUT = "data/processed/ood_wealth_index_clusters.csv"
COV = 0.7   # household feature-coverage threshold (matches dhs.pooled_wealth_index)


def extract_dta(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        dta = next((n for n in z.namelist() if n.upper().endswith(".DTA")), None)
        if not dta: return None
        tmp = Path(tempfile.mkdtemp()); z.extract(dta, tmp); return tmp / dta


def load_surveys(manifest, raw, era=None):
    """Return {cc__year: (features_df, cluster_series)} for HR surveys in the manifest."""
    man = pd.read_csv(manifest); hr = man[man.kind == "HR"]
    if era: hr = hr[(hr.year >= era[0]) & (hr.year <= era[1])]
    feats, clusts = {}, {}
    for _, r in hr.iterrows():
        zp = raw / r.cc / r.filename
        if not zp.exists():
            print(f"  miss {zp.name}"); continue
        try:
            dta = extract_dta(zp)
            if dta is None: print(f"  no DTA {zp.name}"); continue
            df = load_dhs_hr(dta)
            if "hv001" not in df.columns: print(f"  no hv001 {r.cc}{r.year}"); continue
            key = f"{r.cc}__{int(r.year)}"
            feats[key] = extract_asset_features(df).reset_index(drop=True)
            clusts[key] = df["hv001"].reset_index(drop=True)
            print(f"  {key}: {len(feats[key])} hh", flush=True)
        except Exception as e:
            print(f"  ERR {r.cc}{r.year}: {str(e)[:70]}")
    return feats, clusts


def pool_impute(feats, clusts):
    """Pool + drop low-coverage + within-survey-mean impute. Returns (pooled_df, feature_cols)."""
    frames = []
    for k, f in feats.items():
        t = f.copy(); t["country"] = k; t["cluster_id"] = clusts[k].values; frames.append(t)
    pooled = pd.concat(frames, ignore_index=True)
    fcols = [c for c in pooled.columns if c not in {"country", "cluster_id"}]
    pooled = pooled.loc[pooled[fcols].notna().mean(axis=1) >= COV].copy()
    for cc in pooled["country"].unique():
        m = pooled["country"] == cc
        pooled.loc[m, fcols] = pooled.loc[m, fcols].fillna(pooled.loc[m, fcols].mean())
    return pooled, fcols


def main():
    # ---------- 1. fit frozen transform on TRAINING ----------
    print("=== fitting frozen PCA on 23-country training HR ===")
    tf, tc = load_surveys(TRAIN_MANIFEST, TRAIN_RAW, ERA)
    pooled, fcols = pool_impute(tf, tc)
    global_mean = pooled[fcols].mean()                       # frozen fallback for OOD
    pooled[fcols] = pooled[fcols].fillna(global_mean)
    X = pooled[fcols].values.astype(float)
    feat_mean, feat_std = X.mean(0), X.std(0, ddof=0)
    Xs = (X - feat_mean) / feat_std
    pca = PCA(n_components=1); raw = pca.fit_transform(Xs).ravel()
    pc1_mean, pc1_std = raw.mean(), raw.std(ddof=0)
    pc1 = (raw - pc1_mean) / pc1_std
    sign = 1.0
    if "has_electricity" in fcols and pca.components_[0, fcols.index("has_electricity")] < 0:
        sign = -1.0
    pooled["wealth_index"] = sign * pc1

    def project(features_df):
        """Frozen projection of new households -> wealth index on the training axis."""
        f = features_df.reindex(columns=fcols)
        f = f.loc[f.notna().mean(axis=1) >= COV].copy()
        f = f.fillna(f.mean()).fillna(global_mean)           # within-survey then frozen global
        z = (f.values.astype(float) - feat_mean) / feat_std
        return sign * (pca.transform(z).ravel() - pc1_mean) / pc1_std, f.index

    # ---------- 2. validate refit == training labels ----------
    cl = (pooled.groupby(["country", "cluster_id"])["wealth_index"]
          .agg(["mean", "count"]).reset_index()
          .rename(columns={"mean": "wealth_index_mean", "count": "n_households"}))
    kk = cl["country"].str.split("__", expand=True); cl["cc"] = kk[0]; cl["year"] = kk[1].astype(int)
    ref = pd.read_csv(TRAIN_LABELS)
    chk = cl.merge(ref, left_on=["cc", "year", "cluster_id"], right_on=["country", "year", "cluster_id"],
                   suffixes=("_new", "_ref"))
    r = np.corrcoef(chk.wealth_index_mean_new, chk.wealth_index_mean_ref)[0, 1]
    mad = np.abs(chk.wealth_index_mean_new - chk.wealth_index_mean_ref).mean()
    print(f"\n=== VALIDATION: refit vs trained labels  corr={r:.4f}  mean|diff|={mad:.4f}  (want corr~1, diff~0) ===")
    print(f"PC1 explained variance {pca.explained_variance_ratio_[0]*100:.1f}% | sign {sign:+.0f}")
    if r < 0.98:
        print("WARNING: refit diverges from training labels — frozen projector may be off!")

    # ---------- 3. project OOD ----------
    print("\n=== projecting OOD countries through frozen transform ===")
    of, oc = load_surveys(OOD_MANIFEST, OOD_RAW)
    rows = []
    for key in of:
        cc, yr = key.split("__")
        w, idx = project(of[key])
        clusters = oc[key].reindex(idx).values
        d = pd.DataFrame({"cluster_id": clusters, "w": w})
        g = d.groupby("cluster_id")["w"].agg(["mean", "count"]).reset_index()
        for _, rr in g.iterrows():
            rows.append(dict(country=cc, year=int(yr), cluster_id=int(rr.cluster_id),
                             wealth_index_mean=float(rr["mean"]), n_households=int(rr["count"])))
        print(f"  {key}: {len(g)} clusters, wealth [{g['mean'].min():+.2f},{g['mean'].max():+.2f}] "
              f"mean {g['mean'].mean():+.2f}", flush=True)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("ERROR: no OOD clusters produced — HR files unreadable (wrong format?) or no matching columns.")
        return
    out.to_csv(OUT, index=False)
    print(f"\nsaved {len(out)} OOD clusters across {out['country'].nunique()} countries -> {OUT}")
    print("OOD vs training wealth range (training ~[-2.2, 3.0]):")
    print(f"  OOD wealth [{out['wealth_index_mean'].min():+.2f}, {out['wealth_index_mean'].max():+.2f}] "
          f"mean {out['wealth_index_mean'].mean():+.2f}")
    print("  per-country mean wealth (vs training mean ~0):")
    print(out.groupby('country')['wealth_index_mean'].mean().round(2).to_string())


if __name__ == "__main__":
    main()
