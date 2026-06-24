"""Count clusters in the downloaded multi-round DHS GPS files.

Each GE (GPS) shapefile has one point per survey cluster, so its feature count
IS the cluster count for that survey. Reads straight from the zips (no extract).
Gives the total + per-country + per-survey, vs the current single-round 13,453.

Usage: PPY scripts/13_count_clusters.py
"""
from __future__ import annotations

import glob
import zipfile
from collections import defaultdict

import pyogrio

CUR = 13453  # current single-round trainable clusters


def shp_in_zip(z):
    for n in zipfile.ZipFile(z).namelist():
        if n.lower().endswith(".shp"):
            return n
    return None


def main():
    zips = sorted(glob.glob("data/raw/dhs_multiround/*/*GE*.ZIP") +
                  glob.glob("data/raw/dhs_multiround/*/*GE*.zip"))
    per_country = defaultdict(int)
    per_survey = []
    total = 0
    bad = 0
    for z in zips:
        try:
            shp = shp_in_zip(z)
            if not shp:
                bad += 1; continue
            info = pyogrio.read_info(f"/vsizip/{z}/{shp}")
            n = int(info["features"])
            cc = z.split("/")[-2]
            per_country[cc] += n
            per_survey.append((z.split("/")[-1], n))
            total += n
        except Exception as e:
            bad += 1
            print("  bad:", z, str(e)[:60])

    print(f"\n=== GPS files read: {len(zips)-bad} ok, {bad} bad ===")
    print(f"{'cc':3} clusters")
    for cc in sorted(per_country):
        print(f"  {cc:3} {per_country[cc]:>6}")
    print(f"\n>>> TOTAL clusters across downloaded multi-round GPS: {total:,}")
    print(f">>> current single-round (pilot): {CUR:,}")
    print(f">>> multiplier: ~{total/CUR:.1f}x  (NEW clusters if we extract all: ~{total-CUR:,})")
    # rough extraction-time framing (single-round took the EE/Drive multi-day ordeal)
    print(f">>> extraction would be ~{total/CUR:.1f}x the single-round tile pull")


if __name__ == "__main__":
    main()
