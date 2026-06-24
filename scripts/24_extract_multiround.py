"""Bulk-submit Earth Engine exports for the multi-round NEW clusters -> Drive.

Reads data/processed/multiround_extract_targets.csv (the 2008-2022 clusters we
don't already have), submits one 8-channel tile export per cluster to the Drive
folder 'poverty_cnn_multiround'. Resumable via a manifest; quota backpressure is
retried with exponential backoff (the lesson from the first extraction).

ALWAYS validate the bulk path first:  MPY scripts/24_extract_multiround.py --limit 3
Then the full run (unattended):       caffeinate -i MPY scripts/24_extract_multiround.py

cluster_id naming: <country>_<cluster_id>_<survey_year>  (matches the pilot scheme).
"""
from __future__ import annotations

import argparse, csv, time
from pathlib import Path

import pandas as pd
from poverty_cnn.data.earth_engine import export_cluster_to_drive, init_ee

# retry on quota backpressure AND transient network errors (multi-day run must
# survive a DNS/connection blip — the first attempt crashed on one).
_QUOTA_HINTS = ("too many tasks", "quota", "rate limit", "concurrent", "limit",
                "connection", "resolve", "timed out", "timeout", "max retries",
                "temporarily", "unavailable", "503", "502", "reset by peer")
TARGETS = "data/processed/multiround_extract_targets.csv"
MANIFEST = Path("results/multiround_extraction_manifest.csv")
FOLDER = "poverty_cnn_multiround"


def load_submitted():
    if not MANIFEST.exists(): return set()
    with open(MANIFEST, newline="") as f:
        return {r["cluster_id"] for r in csv.DictReader(f)}


def append(row):
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id","lat","lon","year","urban","task_id","timestamp"])
        if new: w.writeheader()
        w.writerow(row)


def submit_retry(lat, lon, year, cid, max_wait=300):
    wait = 15
    while True:
        try:
            return export_cluster_to_drive(lat=lat, lon=lon, year=year,
                                           cluster_id=cid, drive_folder=FOLDER)
        except Exception as e:
            if any(h in str(e).lower() for h in _QUOTA_HINTS):
                print(f"    quota backpressure — waiting {wait}s ({cid})", flush=True)
                time.sleep(wait); wait = min(wait*2, max_wait)
            else:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="submit only first N (validation)")
    args = ap.parse_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing Earth Engine..."); init_ee(".env")
    df = pd.read_csv(TARGETS)
    already = load_submitted()
    print(f"targets: {len(df)} | already submitted: {len(already)} | Drive folder: {FOLDER}")

    submitted = skipped = 0
    for _, r in df.iterrows():
        if args.limit and submitted >= args.limit: break
        cid = f"{r['country']}_{int(r['cluster_id'])}_{int(r['survey_year'])}"
        if cid in already: skipped += 1; continue
        task = submit_retry(float(r["lat"]), float(r["lon"]), int(r["survey_year"]), cid)
        append({"cluster_id": cid, "lat": r["lat"], "lon": r["lon"], "year": int(r["survey_year"]),
                "urban": r.get("urban",""), "task_id": getattr(task.task,"id",""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        submitted += 1
        if submitted % 100 == 0: print(f"  submitted {submitted}...", flush=True)
    print(f"\nDone. Submitted {submitted}, skipped {skipped} already-done.")
    print(f"Manifest: {MANIFEST} | Monitor: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
