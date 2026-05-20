"""Bulk-submit Earth Engine export jobs for a country's DHS clusters → Google Drive.

For each cluster in a DHS GPS shapefile, submits an asynchronous 8-channel
Landsat + nightlights tile export to Google Drive (Yeh 2020 protocol). Submission
is lightweight — Google computes the tiles on its servers and drops them in the
Drive folder. This machine only needs to stay on long enough to finish submitting
and to ride out Earth Engine's task-quota backpressure; once a task is accepted,
it completes on Google's side even if this machine is later turned off.

Resumable: a manifest CSV records every cluster whose export has been submitted,
so re-running skips already-submitted clusters.

Usage:
    # validate on a few clusters first:
    python scripts/03_download_imagery.py --country KE --year 2014 \
        --gps data/raw/dhs/KE/KEGE71FL/KEGE71FL.shp --limit 3

    # full run:
    python scripts/03_download_imagery.py --country KE --year 2014 \
        --gps data/raw/dhs/KE/KEGE71FL/KEGE71FL.shp
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from poverty_cnn.data.dhs import load_dhs_gps
from poverty_cnn.data.earth_engine import export_cluster_to_drive, init_ee

MANIFEST_FIELDS = ["cluster_id", "dhsclust", "lat", "lon", "urban_rural", "task_id", "timestamp"]

# Substrings that indicate EE task-quota backpressure (vs a genuine error).
_QUOTA_HINTS = ("too many", "quota", "limit", "capacity", "rate limit")


def load_submitted(manifest_path: Path) -> set[str]:
    """Return the set of cluster_ids already recorded in the manifest."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as f:
        return {row["cluster_id"] for row in csv.DictReader(f)}


def append_manifest(manifest_path: Path, row: dict) -> None:
    """Append one submitted-cluster record, writing the header if the file is new."""
    new_file = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def submit_with_retry(*, lat, lon, year, cluster_id, drive_folder, max_wait=300):
    """Submit one export, backing off and retrying when EE's task quota is hit.

    Quota backpressure is retried indefinitely with exponential backoff. Any
    other exception is raised immediately so real bugs surface during validation.
    """
    wait = 15
    while True:
        try:
            return export_cluster_to_drive(
                lat=lat, lon=lon, year=year,
                cluster_id=cluster_id, drive_folder=drive_folder,
            )
        except Exception as e:  # noqa: BLE001 — inspect message to classify
            msg = str(e).lower()
            if any(hint in msg for hint in _QUOTA_HINTS):
                print(f"    quota backpressure — waiting {wait}s before retrying {cluster_id}")
                time.sleep(wait)
                wait = min(wait * 2, max_wait)
            else:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit DHS-cluster 8-channel tile exports to Google Drive via Earth Engine."
    )
    parser.add_argument("--country", required=True, help="Country code, e.g. KE")
    parser.add_argument("--year", type=int, default=None,
                        help="DHS survey year. If omitted, read per-cluster from the GPS DHSYEAR field (recommended).")
    parser.add_argument("--gps", required=True, help="Path to the DHS GPS .shp file")
    parser.add_argument("--drive-folder", default="poverty_cnn_data", help="Google Drive output folder")
    parser.add_argument("--manifest", default=None,
                        help="Manifest CSV path (default: results/extraction_manifest_<country>.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only submit the first N valid clusters (for validation)")
    args = parser.parse_args()

    manifest_path = (Path(args.manifest) if args.manifest
                     else Path(f"results/extraction_manifest_{args.country}.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing Earth Engine...")
    init_ee()

    print(f"Loading GPS: {args.gps}")
    gdf = load_dhs_gps(args.gps)
    print(f"  {len(gdf)} clusters in file")

    already = load_submitted(manifest_path)
    if already:
        print(f"  {len(already)} clusters already submitted (manifest) — will skip them")

    submitted = skipped_zero = skipped_done = 0
    for _, row in gdf.iterrows():
        dhsclust = int(row["DHSCLUST"])
        lat = float(row["LATNUM"])
        lon = float(row["LONGNUM"])
        urban = str(row.get("URBAN_RURA", ""))

        # DHS encodes clusters with missing GPS as (0, 0) — skip them.
        if lat == 0.0 and lon == 0.0:
            skipped_zero += 1
            continue

        # Survey year: explicit --year if given, otherwise read it straight
        # from the GPS DHSYEAR field. Reading from the data avoids relying on a
        # caller-supplied year map (which silently broke once on macOS bash 3.2).
        year = args.year if args.year is not None else int(row["DHSYEAR"])

        cluster_id = f"{args.country}_{dhsclust}_{year}"
        if cluster_id in already:
            skipped_done += 1
            continue

        task = submit_with_retry(
            lat=lat, lon=lon, year=year,
            cluster_id=cluster_id, drive_folder=args.drive_folder,
        )
        append_manifest(manifest_path, {
            "cluster_id": cluster_id,
            "dhsclust": dhsclust,
            "lat": lat,
            "lon": lon,
            "urban_rural": urban,
            "task_id": getattr(task.task, "id", ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        submitted += 1
        if submitted % 50 == 0:
            print(f"  submitted {submitted}...")

        if args.limit and submitted >= args.limit:
            print(f"  reached --limit {args.limit}, stopping")
            break

    print(f"\nDone. Submitted {submitted} new exports "
          f"(skipped {skipped_zero} zero-GPS, {skipped_done} already-done).")
    print(f"Manifest:   {manifest_path}")
    print(f"Drive folder: {args.drive_folder}")
    print("Monitor tasks at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
