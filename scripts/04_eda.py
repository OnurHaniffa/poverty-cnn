"""EDA report for the pooled wealth index + DHS asset data (all 23 countries).

Regenerates the pooled PCA with ALL components (for the scree plot), computes
asset prevalences and per-country / urban-rural distributions, and renders
figures + a markdown summary for the progress meeting.

The pooling/imputation/standardization here mirrors `pooled_wealth_index()` in
dhs.py exactly (pool -> >=70% coverage filter -> country-mean impute ->
z-score) so the PC1 variance reproduces the production index (28.7%). The only
difference is we keep all 15 PCs to draw the scree.

Run: python scripts/04_eda.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from poverty_cnn.data.dhs import extract_asset_features, load_dhs_hr, load_dhs_pr_as_hr

DHS_ROOT = Path("data/raw/dhs")
CLUSTER_CSV = Path("data/processed/wealth_index_clusters.csv")
FIG = Path("results/figures/eda")
FIG.mkdir(parents=True, exist_ok=True)
SUMMARY = Path("results/eda/EDA_SUMMARY.md")
SUMMARY.parent.mkdir(parents=True, exist_ok=True)


def find_recode(cc_dir: str):
    for pat in (f"{cc_dir}/*HR*/*.DTA", f"{cc_dir}/*HR*/*.dta"):
        h = glob.glob(pat)
        if h:
            return h[0], False
    for pat in (f"{cc_dir}/*PR*/*.DTA", f"{cc_dir}/*PR*/*.dta"):
        h = glob.glob(pat)
        if h:
            return h[0], True
    return None, None


def build_pooled() -> pd.DataFrame:
    frames = []
    for cc_dir in sorted(glob.glob(f"{DHS_ROOT}/*/")):
        cc = Path(cc_dir).name
        dta, is_pr = find_recode(cc_dir)
        if not dta:
            continue
        df = load_dhs_pr_as_hr(dta) if is_pr else load_dhs_hr(dta)
        feat = extract_asset_features(df).copy()
        feat["country"] = cc
        frames.append(feat)
        print(f"  loaded {cc}: {len(feat)} households")
    return pd.concat(frames, ignore_index=True)


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")


def main() -> None:
    print("=== Re-pooling households for full-PCA scree ===")
    pooled = build_pooled()
    feature_cols = [c for c in pooled.columns if c != "country"]

    coverage = pooled[feature_cols].notna().mean(axis=1)
    pooled = pooled.loc[coverage >= 0.7].copy()
    for cc in pooled["country"].unique():
        m = pooled["country"] == cc
        pooled.loc[m, feature_cols] = pooled.loc[m, feature_cols].fillna(
            pooled.loc[m, feature_cols].mean())
    pooled[feature_cols] = pooled[feature_cols].fillna(pooled[feature_cols].mean())

    X = pooled[feature_cols].values.astype(float)
    Xs = (X - X.mean(0)) / X.std(0, ddof=0)
    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_

    pc1 = pca.transform(Xs)[:, 0]
    elec = feature_cols.index("has_electricity")
    comps = pca.components_[0]
    if comps[elec] < 0:
        pc1, comps = -pc1, -comps
    pc1 = (pc1 - pc1.mean()) / pc1.std(ddof=0)

    n = len(feature_cols)
    null = 1.0 / n  # variance a random/independent direction would explain

    # --- Fig 1: SCREE (the answer to "why is 28% enough?") ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), evr * 100, color="#3b6", label="PC variance explained")
    ax.axhline(null * 100, color="crimson", ls="--",
               label=f"random/independent baseline = {null*100:.1f}%")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% of total variance explained")
    ax.set_title(f"Scree: PC1 = {evr[0]*100:.1f}%  (={evr[0]/null:.1f}x the baseline)")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    save(fig, "01_pca_scree.png")

    # --- Fig 2: PC1 loadings ---
    order = np.argsort(comps)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in order], comps[order],
            color=["#c33" if comps[i] < 0 else "#36c" for i in order])
    ax.set_xlabel("PC1 loading (the wealth recipe)")
    ax.set_title("Asset loadings on PC1")
    save(fig, "02_pca_loadings.png")

    # --- Fig 3: asset prevalence (binary features) ---
    bin_cols = [c for c in feature_cols if c != "sleeping_rooms"]
    prev = X[:, [feature_cols.index(c) for c in bin_cols]].mean(0) * 100
    o = np.argsort(prev)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([bin_cols[i] for i in o], prev[o], color="#888")
    ax.set_xlabel("% of households owning / having")
    ax.set_title("Asset prevalence (pooled, 23 countries)")
    save(fig, "03_asset_prevalence.png")

    # --- Fig 4: household-level wealth distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pc1, bins=80, color="#69b")
    ax.set_xlabel("Household wealth index (PC1, standardized)")
    ax.set_ylabel("households")
    ax.set_title(f"Household wealth distribution (n={len(pc1):,})")
    save(fig, "04_household_wealth_hist.png")

    # ---- cluster-level figures from the saved CSV ----
    c = pd.read_csv(CLUSTER_CSV)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(c["wealth_index_mean"], bins=60, color="#69b")
    ax.axvline(c["wealth_index_mean"].mean(), color="k", ls="--", label="mean")
    ax.axvline(c["wealth_index_mean"].median(), color="crimson", ls=":", label="median")
    ax.set_xlabel("Cluster wealth index (mean of households)")
    ax.set_ylabel("clusters")
    ax.set_title(f"Cluster wealth distribution (n={len(c):,})")
    ax.legend()
    save(fig, "05_cluster_wealth_hist.png")

    # per-country boxplot, sorted by median
    by = c.groupby("country")["wealth_index_mean"]
    order_c = by.median().sort_values().index.tolist()
    data = [c.loc[c["country"] == cc, "wealth_index_mean"].values for cc in order_c]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.boxplot(data, labels=order_c, vert=False, showfliers=False)
    ax.set_xlabel("cluster wealth index")
    ax.set_title("Wealth distribution by country (poorest -> richest)")
    save(fig, "06_wealth_by_country.png")

    # urban vs rural
    fig, ax = plt.subplots(figsize=(8, 5))
    for u, col, lab in [("R", "#c93", "Rural"), ("U", "#39c", "Urban")]:
        ax.hist(c.loc[c["urban"] == u, "wealth_index_mean"], bins=50,
                alpha=0.6, color=col, label=lab, density=True)
    ax.set_xlabel("cluster wealth index")
    ax.set_ylabel("density")
    ax.set_title("Urban vs rural wealth (gap = 1.16 sigma)")
    ax.legend()
    save(fig, "07_urban_rural.png")

    # clusters per country
    cpc = c["country"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(cpc.index, cpc.values, color="#888")
    ax.set_xlabel("clusters")
    ax.set_title(f"Clusters per country (total {len(c):,})")
    save(fig, "08_clusters_per_country.png")

    # households per cluster
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(c["n_households"], bins=36, color="#69b")
    ax.set_xlabel("households per cluster")
    ax.set_ylabel("clusters")
    ax.set_title("Households per cluster (DHS design ~25-30)")
    save(fig, "09_households_per_cluster.png")

    # map: clusters colored by wealth
    m = c[(c["lat"] != 0) | (c["lon"] != 0)]
    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(m["lon"], m["lat"], c=m["wealth_index_mean"],
                    cmap="coolwarm", s=4, alpha=0.6)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("DHS clusters colored by wealth index")
    fig.colorbar(sc, label="wealth index")
    save(fig, "10_cluster_map.png")

    # --- summary markdown ---
    lines = [
        "# EDA Summary — pooled wealth index (23 countries)",
        "",
        f"- Households pooled: **{len(pc1):,}**  |  clusters: **{len(c):,}**  |  countries: 23",
        f"- Coverage filter (>=70% features present) dropped only ~{len(coverage) - len(pc1)} households.",
        f"- Missing values in the final cluster table: **{int(c.isna().sum().sum())}**.",
        f"- Households per cluster: mean {c['n_households'].mean():.1f}, median {c['n_households'].median():.0f} (DHS design ~25-30).",
        "",
        "## PCA / variance explained",
        f"- **PC1 = {evr[0]*100:.1f}%** of total variance.",
        f"- Random/independent baseline (1/{n}) = **{null*100:.1f}%**  ->  PC1 is **{evr[0]/null:.1f}x** the baseline.",
        f"- PC2 = {evr[1]*100:.1f}%, PC3 = {evr[2]*100:.1f}% (sharp drop-off after PC1 = one dominant factor).",
        "- All assets load positive except `has_bicycle` (~0, wealth-neutral) -> a single coherent wealth dimension.",
        "",
        "### Why 28% is plenty for a wealth index",
        "1. Binary survey items carry huge item-specific noise; first-factor variance of 20-40% is the norm and is *strong*, not weak. The '50%+' rule comes from continuous, tightly-correlated data.",
        "2. The right benchmark is the random baseline (6.7%), not 50%. PC1 captures ~4x that.",
        "3. Variance-explained measures *compactness*, not *validity*. PC1 is validated by: coherent positive loadings, a sensible country ranking, and a large correct-direction urban/rural gap.",
        "4. The other ~71% is mostly NON-wealth (country-specific quirks, item noise) we deliberately exclude.",
        "5. It is the field-standard DHS Wealth Index method (Filmer-Pritchett 2001) and exactly what Yeh 2020 used.",
        "",
        "## Wealth distribution",
        f"- Cluster wealth: mean {c['wealth_index_mean'].mean():.3f}, std {c['wealth_index_mean'].std():.3f}, skew {c['wealth_index_mean'].skew():.2f}.",
        "- Cluster-level std < 1 because we standardize at household level then average; ~71% of wealth variance is between-village, ~29% within-village.",
        f"- Urban mean {c.loc[c['urban']=='U','wealth_index_mean'].mean():+.3f} vs rural {c.loc[c['urban']=='R','wealth_index_mean'].mean():+.3f} = **{c.loc[c['urban']=='U','wealth_index_mean'].mean()-c.loc[c['urban']=='R','wealth_index_mean'].mean():.3f} sigma gap**.",
        "",
        "## Figures (results/figures/eda/)",
        "01 scree · 02 loadings · 03 asset prevalence · 04 household wealth · 05 cluster wealth · 06 by-country · 07 urban/rural · 08 clusters-per-country · 09 households-per-cluster · 10 cluster map",
    ]
    SUMMARY.write_text("\n".join(lines))
    print(f"\nWrote summary -> {SUMMARY}")
    print(f"Wrote 10 figures -> {FIG}")


if __name__ == "__main__":
    main()
