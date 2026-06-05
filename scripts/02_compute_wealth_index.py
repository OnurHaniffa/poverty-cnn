"""Compute the pooled asset wealth index across all 23 DHS countries.

For each country: load the Household Recode (or Person Recode, deduped, for
Kenya), extract asset features, and collect the per-household cluster IDs.
Then pool every household, run PCA, take PC1 as the household wealth index,
average to cluster level, and join urban/rural + GPS. The output CSV is the
CNN's regression target table.

Run: python scripts/02_compute_wealth_index.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from poverty_cnn.data.dhs import (
    extract_asset_features,
    load_dhs_gps,
    load_dhs_hr,
    load_dhs_pr_as_hr,
    pooled_wealth_index,
)

DHS_ROOT = Path("data/raw/dhs")
OUT = Path("data/processed/wealth_index_clusters.csv")


def find_recode(cc_dir: str):
    """Return (path, is_pr) for the country's HR (preferred) or PR .DTA."""
    for pat in (f"{cc_dir}/*HR*/*.DTA", f"{cc_dir}/*HR*/*.dta"):
        hits = glob.glob(pat)
        if hits:
            return hits[0], False
    for pat in (f"{cc_dir}/*PR*/*.DTA", f"{cc_dir}/*PR*/*.dta"):
        hits = glob.glob(pat)
        if hits:
            return hits[0], True
    return None, None


def main() -> None:
    country_features: dict[str, pd.DataFrame] = {}
    country_clusters: dict[str, pd.Series] = {}
    gps_by_country: dict[str, "pd.DataFrame"] = {}

    print("=== Loading per-country survey + GPS ===")
    for cc_dir in sorted(glob.glob(f"{DHS_ROOT}/*/")):
        cc = Path(cc_dir).name
        dta, is_pr = find_recode(cc_dir)
        if not dta:
            print(f"  {cc}: no recode .DTA found — SKIP")
            continue
        try:
            df = load_dhs_pr_as_hr(dta) if is_pr else load_dhs_hr(dta)
            feats = extract_asset_features(df)
            country_features[cc] = feats
            country_clusters[cc] = df["hv001"].reset_index(drop=True)
            shp = glob.glob(f"{cc_dir}/*GE*/*.shp")
            if shp:
                gps_by_country[cc] = load_dhs_gps(shp[0])
            print(f"  {cc}: {len(df):>6} households ({'PR' if is_pr else 'HR'}), "
                  f"{feats.shape[1]} features, GPS={'yes' if shp else 'NO'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {cc}: ERROR — {e} — SKIP")

    print(f"\n=== Pooling {len(country_features)} countries, running PCA ===")
    result = pooled_wealth_index(country_features, country_clusters)
    print(f"Households pooled:        {len(result.household_index):,}")
    print(f"Clusters:                 {len(result.cluster_index):,}")
    print(f"PC1 variance explained:   {result.explained_variance_ratio:.1%}")
    print("PC1 loadings (by |weight|):")
    for name, load in sorted(zip(result.feature_names, result.pca_components),
                             key=lambda x: -abs(x[1])):
        print(f"   {name:18s} {load:+.3f}")

    # Join GPS (urban/rural, lat/lon, year) to the cluster table.
    ci = result.cluster_index.copy()
    ci["cluster_id"] = ci["cluster_id"].astype(int)
    rows = []
    for cc, g in gps_by_country.items():
        for _, r in g.iterrows():
            rows.append({
                "country": cc,
                "cluster_id": int(r["DHSCLUST"]),
                "urban": str(r.get("URBAN_RURA", "")),
                "lat": float(r["LATNUM"]),
                "lon": float(r["LONGNUM"]),
                "year": int(r["DHSYEAR"]) if "DHSYEAR" in g.columns else None,
            })
    gps_df = pd.DataFrame(rows)
    merged = ci.merge(gps_df, on=["country", "cluster_id"], how="left")

    u = merged.loc[merged["urban"] == "U", "wealth_index_mean"].mean()
    r = merged.loc[merged["urban"] == "R", "wealth_index_mean"].mean()
    print(f"\nUrban mean wealth: {u:+.3f} | Rural mean: {r:+.3f} | gap: {u - r:.3f} sigma")
    print(f"Clusters with GPS: {merged['lat'].notna().sum():,} / {len(merged):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}  ({len(merged):,} clusters)")


if __name__ == "__main__":
    main()
