"""Submit Earth Engine tile exports for the OOD clusters -> Google Drive.

Reads data/processed/ood_extract_targets.csv and submits one 8-channel tile export per
cluster to the Drive folder 'poverty_cnn_ood'. Resumable via a manifest (already-submitted
clusters are skipped); quota/network backpressure is retried with exponential backoff (the
multi-hour run must survive a blip). cluster_id naming: <country>_<cluster>_<year>.

Validate first:  PPY scripts/40_ood_extract.py --limit 3
Full unattended:  PPY scripts/40_ood_extract.py    (run detached on the PC)
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import pandas as pd
from poverty_cnn.data.earth_engine import export_cluster_to_drive, init_ee

_QUOTA_HINTS = ("too many tasks", "quota", "rate limit", "concurrent", "limit",
                "connection", "resolve", "timed out", "timeout", "max retries",
                "temporarily", "unavailable", "503", "502", "reset by peer")
TARGETS = "data/processed/ood_extract_targets.csv"
MANIFEST = Path("results/ood_extraction_manifest.csv")
FOLDER = "poverty_cnn_ood"


def already_done():
    if not MANIFEST.exists():
        return set()
    return set(pd.read_csv(MANIFEST)["cluster_id"].astype(str))


def append(row):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "lat", "lon", "year", "urban", "task_id", "timestamp"])
        if new:
            w.writeheader()
        w.writerow(row)


def submit_retry(lat, lon, year, cid, max_wait=300):
    wait = 10
    while True:
        try:
            return export_cluster_to_drive(lat=lat, lon=lon, year=year, cluster_id=cid, drive_folder=FOLDER)
        except Exception as e:
            if any(h in str(e).lower() for h in _QUOTA_HINTS):
                print(f"    backpressure ({str(e)[:50]}), sleep {wait}s", flush=True)
                time.sleep(wait); wait = min(max_wait, wait * 2); continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    init_ee()
    df = pd.read_csv(TARGETS)
    if args.limit:
        df = df.head(args.limit)
    done = already_done()
    print(f"OOD extraction: {len(df)} targets, {len(done)} already submitted -> Drive '{FOLDER}'")
    n = 0
    for _, r in df.iterrows():
        cid = f"{r['country']}_{int(r['cluster_id'])}_{int(r['survey_year'])}"
        if cid in done:
            continue
        task = submit_retry(float(r["lat"]), float(r["lon"]), int(r["survey_year"]), cid)
        append({"cluster_id": cid, "lat": r["lat"], "lon": r["lon"], "year": int(r["survey_year"]),
                "urban": r["urban"], "task_id": getattr(task, "task_id", ""), "timestamp": int(time.time())})
        n += 1
        if n % 50 == 0:
            print(f"  submitted {n} ({cid})", flush=True)
        time.sleep(0.4)   # gentle pacing under the concurrent-task ceiling
    print(f"\ndone: submitted {n} new exports ({len(done)+n}/{len(df)} total) -> Drive '{FOLDER}'")
    print("Google now renders the tiles server-side (hours). Next: rclone download -> cache -> frozen test.")


if __name__ == "__main__":
    main()
