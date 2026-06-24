"""Build the OOD extraction-targets list: join OOD cluster GPS to the frozen-PCA wealth.

Reads each OOD country's GE (GPS) shapefile straight from the zip, drops missing-GPS
(Null Island / SOURCE=MIS), and inner-joins to data/processed/ood_wealth_index_clusters.csv
(only clusters with BOTH a GPS point and a projected wealth label survive). Output drives
the Earth Engine tile extraction and, later, the frozen-model OOD test.

Output: data/processed/ood_extract_targets.csv
  columns: country, cluster_id, survey_year, lat, lon, urban, wealth_index_mean

Run on PC: PPY scripts/39_ood_gps_targets.py
"""
from __future__ import annotations
import glob, zipfile
import numpy as np
import pandas as pd
import pyogrio

WEALTH = "data/processed/ood_wealth_index_clusters.csv"
OUT = "data/processed/ood_extract_targets.csv"


def shp_in_zip(z):
    return next((n for n in zipfile.ZipFile(z).namelist() if n.lower().endswith(".shp")), None)


def main():
    zips = sorted(glob.glob("data/raw/dhs_ood/*/*GE*.ZIP") + glob.glob("data/raw/dhs_ood/*/*GE*.zip"))
    rows, bad = [], 0
    for z in zips:
        cc = z.split("/")[-2]
        shp = shp_in_zip(z)
        if not shp:
            bad += 1; continue
        try:
            gdf = pyogrio.read_dataframe(f"/vsizip/{z}/{shp}")
        except Exception as e:
            bad += 1; print("  bad:", z, str(e)[:50]); continue
        cols = {c.upper(): c for c in gdf.columns}
        sub = pd.DataFrame({
            "country": cc,
            "cluster_id": gdf[cols["DHSCLUST"]].astype(int),
            "survey_year": gdf[cols["DHSYEAR"]].astype(int) if "DHSYEAR" in cols else np.nan,
            "lat": gdf[cols["LATNUM"]].astype(float),
            "lon": gdf[cols["LONGNUM"]].astype(float),
            "urban": gdf[cols["URBAN_RURA"]] if "URBAN_RURA" in cols else "",
            "source": gdf[cols["SOURCE"]] if "SOURCE" in cols else "",
        })
        rows.append(sub)
        print(f"  {cc}: {len(sub)} GPS points")
    gps = pd.concat(rows, ignore_index=True)
    n0 = len(gps)
    gps = gps[~((gps.lat.abs() < 1e-6) & (gps.lon.abs() < 1e-6)) & (gps.source.astype(str) != "MIS")]
    print(f"\nGE files: {len(zips)-bad} ok, {bad} bad | GPS {n0} -> {len(gps)} valid (dropped {n0-len(gps)} null-island)")

    wealth = pd.read_csv(WEALTH)   # country, year, cluster_id, wealth_index_mean, n_households
    m = gps.merge(wealth[["country", "cluster_id", "wealth_index_mean", "n_households"]],
                  on=["country", "cluster_id"], how="inner")
    m["urban"] = m["urban"].astype(str).str[0].str.upper().map({"U": "U", "R": "R"}).fillna("R")
    out = m[["country", "cluster_id", "survey_year", "lat", "lon", "urban", "wealth_index_mean"]].copy()
    out.to_csv(OUT, index=False)
    print(f"\njoined targets: {len(out)} clusters (GPS ∩ wealth) across {out.country.nunique()} countries")
    print(out.groupby("country").agg(n=("cluster_id", "size"),
          urban=("urban", lambda s: (s == "U").mean())).round(2).to_string())
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
