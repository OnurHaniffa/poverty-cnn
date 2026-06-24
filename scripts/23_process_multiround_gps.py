"""Build the multi-round GPS inventory from the downloaded DHS GE (GPS) zips.

Each GE shapefile has one point per cluster. We read them all (straight from the
zips), drop the missing-GPS 'Null Island' (0,0)/SOURCE=MIS clusters, and build one
table of every (country, survey_year, cluster, lat, lon, urban) we could extract
satellite tiles for. Then report per-era counts so we can pick the extraction
subset (e.g. 2008-2022, good-imagery era, ~Yeh scale).

Run on the Mac (zips are here): MPY scripts/23_process_multiround_gps.py
"""
from __future__ import annotations

import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio


def shp_in_zip(z):
    for n in zipfile.ZipFile(z).namelist():
        if n.lower().endswith(".shp"):
            return n
    return None


def main():
    zips = sorted(glob.glob("data/raw/dhs_multiround/*/*GE*.ZIP") +
                  glob.glob("data/raw/dhs_multiround/*/*GE*.zip"))
    rows = []
    bad = 0
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
        lat = gdf[cols.get("LATNUM")]; lon = gdf[cols.get("LONGNUM")]
        yr = gdf[cols.get("DHSYEAR")] if "DHSYEAR" in cols else np.nan
        urb = gdf[cols.get("URBAN_RURA")] if "URBAN_RURA" in cols else ""
        src = gdf[cols.get("SOURCE")] if "SOURCE" in cols else ""
        clust = gdf[cols.get("DHSCLUST")]
        sub = pd.DataFrame({"country": cc, "survey_year": yr, "cluster_id": clust,
                            "lat": lat, "lon": lon, "urban": urb, "source": src})
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    n0 = len(df)
    # drop missing GPS (Null Island / SOURCE=MIS / near-zero coords)
    valid = ~((df.lat.abs() < 1e-6) & (df.lon.abs() < 1e-6)) & (df.source.astype(str) != "MIS")
    df = df[valid].reset_index(drop=True)
    df.to_csv("data/processed/multiround_gps.csv", index=False)
    print(f"GE files read: {len(zips)-bad} ok, {bad} bad")
    print(f"clusters: {n0} total -> {len(df)} valid GPS (dropped {n0-len(df)} missing-GPS)")
    print(f"countries: {df.country.nunique()} | surveys: {df.groupby(['country','survey_year']).ngroups}")
    print("\n=== clusters per ERA (to pick the extraction subset) ===")
    bins = [(2003,2007),(2008,2012),(2013,2017),(2018,2024)]
    for lo,hi in bins:
        m = (df.survey_year>=lo)&(df.survey_year<=hi)
        print(f"  {lo}-{hi}: {m.sum():6} clusters  ({df[m].groupby(['country','survey_year']).ngroups} surveys)")
    good = (df.survey_year>=2008)&(df.survey_year<=2022)
    print(f"\n  >>> 2008-2022 (good-imagery, VIIRS+L7/L8 era): {good.sum()} clusters")
    print(f"      current single-round pilot: 13453  |  this subset is ~{good.sum()/13453:.1f}x")
    print("saved -> data/processed/multiround_gps.csv")


if __name__ == "__main__":
    main()
