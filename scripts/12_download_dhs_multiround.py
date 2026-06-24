"""Batch-download approved DHS datasets (multi-round) using your logged-in session.

DHS has no public download API — the file fetch needs your authenticated session.
This script reads your browser's exported cookies (credential-safe: no password
ever touches the script) and pulls every HR (Household Recode) + GE (GPS) zip in
the manifest, resumably, validating each is a real zip (not a login-redirect).

ONE-TIME SETUP (you do this in your browser):
  1. Log into https://dhsprogram.com  (your approved account).
  2. Install a "Get cookies.txt LOCALLY" extension (Chrome/Firefox).
  3. Export cookies for dhsprogram.com -> save as cookies.txt.

THEN (validate on ONE file before the full batch):
  PPY scripts/12_download_dhs_multiround.py --cookies cookies.txt --limit 1
  # inspect: did it save a real .zip? if yes ->
  PPY scripts/12_download_dhs_multiround.py --cookies cookies.txt

Downloads land in data/raw/dhs_multiround/<CC>/<filename>.zip
"""
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import random
import time
import zipfile
from pathlib import Path

import requests

# DHS authenticated dataset-download endpoint (ColdFusion legacy path).
# If this pattern 401s/returns HTML for your account, grab the real URL from your
# browser's network tab on a manual download and set --url-template.
# Tp=1 for survey recodes (HR), Tp=2 for GPS (GE). Verified from the DHS dataset
# page HTML (the only difference between a working HR link and a GPS link).
DEFAULT_URL = ("https://dhsprogram.com/customcf/legacy/data/download_dataset.cfm"
               "?Filename={filename}&Tp={tp}&Ctry_Code={cc}&surv_id=0")


def looks_like_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/dhs_multiround_manifest.csv")
    ap.add_argument("--cookies", required=True, help="cookies.txt exported from your browser")
    ap.add_argument("--out", default="data/raw/dhs_multiround")
    ap.add_argument("--url-template", default=DEFAULT_URL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; use 1 to validate first")
    ap.add_argument("--kind", choices=["HR", "GE", "both"], default="both",
                    help="download only Household (HR) or GPS (GE) files, or both")
    ap.add_argument("--sleep-min", type=float, default=3.0, help="min polite delay between files (s)")
    ap.add_argument("--sleep-max", type=float, default=9.0, help="max polite delay between files (s)")
    ap.add_argument("--long-pause-every", type=int, default=25,
                    help="every N files, take a longer 30-70s breather (looks human, avoids rate caps)")
    args = ap.parse_args()

    cj = http.cookiejar.MozillaCookieJar(args.cookies)
    cj.load(ignore_discard=True, ignore_expires=True)
    sess = requests.Session()
    sess.cookies = cj
    sess.headers["User-Agent"] = "Mozilla/5.0"
    # FIX: some browser cookie exports write host-only cookies (domain WITHOUT a leading
    # dot, e.g. "dhsprogram.com") that http.cookiejar then silently refuses to send — so the
    # critical JSESSIONID never goes out and every request is anonymous. Send an explicit
    # Cookie header built from ALL loaded cookies to guarantee the session is transmitted.
    cookie_hdr = "; ".join(f"{c.name}={c.value}" for c in cj)
    if cookie_hdr:
        sess.headers["Cookie"] = cookie_hdr

    rows = list(csv.DictReader(open(args.manifest)))
    if args.kind != "both":
        rows = [r for r in rows if r["kind"] == args.kind]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    ok = skip = fail = 0
    fetched = 0
    for r in rows:
        cc, fn = r["cc"], r["filename"]
        dest = out / cc / fn
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and looks_like_zip(dest):
            skip += 1
            continue
        # polite throttle: random delay before each actual fetch, longer breather periodically
        if fetched:
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
            if args.long_pause_every and fetched % args.long_pause_every == 0:
                pause = random.uniform(30, 70)
                print(f"  ...breather {pause:.0f}s (polite, after {fetched} fetches)")
                time.sleep(pause)
        fetched += 1
        tp = 2 if r["kind"] == "GE" else 1     # GPS=2, survey recode=1
        url = args.url_template.format(filename=fn, cc=cc, tp=tp)
        try:
            resp = sess.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
            if looks_like_zip(dest):
                ok += 1
                print(f"  OK   {cc}/{fn}  ({dest.stat().st_size//1024} KB)")
            else:
                fail += 1
                head = dest.read_bytes()[:200]
                dest.rename(dest.with_suffix(".zip.badhtml"))
                print(f"  FAIL {cc}/{fn} -> not a zip (likely login page). First bytes:\n    {head[:160]!r}")
        except Exception as e:
            fail += 1
            print(f"  ERR  {cc}/{fn}: {e}")
    print(f"\ndone: {ok} downloaded, {skip} already had, {fail} failed (of {len(rows)})")
    if fail and not ok:
        print("ALL failed -> the URL pattern or cookies are wrong. Grab the real download "
              "URL from your browser network tab and pass it via --url-template.")


if __name__ == "__main__":
    main()
