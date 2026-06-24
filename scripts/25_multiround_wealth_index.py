"""Multi-round pooled wealth index for the 'full' dataset (build-ahead).

Refits the pooled-PCA asset wealth index over ALL households from the 2008-2022
survey rounds (the extraction era). One pooled PCA = one comparable 'ruler' across
countries AND time (needed for the temporal-drift analysis; matches Yeh's pooled
composite approach). Asset-drift across years is a documented limitation.

Reuses the validated single-round logic (dhs.py). Reads HR (household recode) Stata
files straight from the downloaded zips. Each survey (country, year) is treated as a
unit so imputation is within-survey while standardization+PCA are global.

Output: data/processed/multiround_wealth_index_clusters.csv
  columns: country, year, cluster_id, wealth_index_mean, n_households, urban, lat, lon

Run on the Mac (zips + GPS here): MPY scripts/25_multiround_wealth_index.py
"""
from __future__ import annotations

import tempfile, zipfile
from pathlib import Path

import pandas as pd

from poverty_cnn.data.dhs import (extract_asset_features, load_dhs_hr,
                                  pooled_wealth_index)

MANIFEST = "data/raw/dhs_multiround_manifest.csv"
GPS = "data/processed/multiround_gps.csv"
RAW = Path("data/raw/dhs_multiround")
ERA = (2008, 2022)
OUT = "data/processed/multiround_wealth_index_clusters.csv"


def extract_dta(zip_path: Path) -> Path | None:
    """Extract the .DTA from a DHS HR zip to a temp dir; return its path."""
    with zipfile.ZipFile(zip_path) as z:
        dta = next((n for n in z.namelist() if n.upper().endswith(".DTA")), None)
        if not dta:
            return None
        tmp = Path(tempfile.mkdtemp())
        z.extract(dta, tmp)
        return tmp / dta


def main():
    man = pd.read_csv(MANIFEST)
    hr = man[(man.kind == "HR") & (man.year >= ERA[0]) & (man.year <= ERA[1])]
    print(f"HR surveys in {ERA[0]}-{ERA[1]}: {len(hr)}")

    country_features, country_clusters = {}, {}
    n_hh = 0
    for _, r in hr.iterrows():
        key = f"{r.cc}__{int(r.year)}"
        zp = RAW / r.cc / r.filename
        if not zp.exists():
            print(f"  miss zip {zp.name}"); continue
        try:
            dta = extract_dta(zp)
            if dta is None:
                print(f"  no .DTA in {zp.name}"); continue
            df = load_dhs_hr(dta)
            if "hv001" not in df.columns:
                print(f"  no hv001 in {key}"); continue
            feats = extract_asset_features(df).reset_index(drop=True)
            clusters = df["hv001"].reset_index(drop=True)
            country_features[key] = feats
            country_clusters[key] = clusters
            n_hh += len(feats)
            print(f"  {key}: {len(feats)} households")
        except Exception as e:
            print(f"  ERR {key}: {str(e)[:70]}")

    print(f"\nPooling {len(country_features)} surveys, {n_hh} households -> PCA")
    res = pooled_wealth_index(country_features, country_clusters)
    ci = res.cluster_index.copy()
    # split the synthetic country__year key back into country + year
    keyed = ci["country"].str.split("__", expand=True)
    ci["country"] = keyed[0]
    ci["year"] = keyed[1].astype(int)

    # join GPS (urban, lat, lon) on country+year+cluster
    gps = pd.read_csv(GPS)[["country", "survey_year", "cluster_id", "urban", "lat", "lon"]]
    gps = gps.rename(columns={"survey_year": "year"})
    out = ci.merge(gps, on=["country", "year", "cluster_id"], how="left")
    out = out[["country", "year", "cluster_id", "wealth_index_mean", "n_households",
               "urban", "lat", "lon"]]
    out.to_csv(OUT, index=False)

    print(f"\nPC1 explained variance: {res.explained_variance_ratio*100:.1f}%")
    print("top PC1 loadings:")
    load = pd.Series(res.pca_components, index=res.feature_names).sort_values(ascending=False)
    print(load.head(6).round(3).to_string())
    print(f"\nclusters: {len(out)} | with GPS: {out.lat.notna().sum()} | "
          f"countries: {out.country.nunique()} | surveys: {out.groupby(['country','year']).ngroups}")
    print(f"wealth range [{out.wealth_index_mean.min():.2f}, {out.wealth_index_mean.max():.2f}] "
          f"mean {out.wealth_index_mean.mean():.3f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
