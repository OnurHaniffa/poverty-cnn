"""Download the OOD tiles from Google Drive (folder poverty_cnn_ood) WITHOUT rclone,
using the Earth Engine OAuth credentials (which carry the Drive scope, since EE exports
to Drive). Resumable: skips files already present in data/raw/landsat_ood/.

Validate:  PPY scripts/41_ood_drive_download.py --limit 5
Full run:  PPY scripts/41_ood_drive_download.py   (run detached on the PC)
"""
from __future__ import annotations
import argparse, io, time
from pathlib import Path
import ee
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from poverty_cnn.data.earth_engine import init_ee

FOLDER = "poverty_cnn_ood"
OUT = Path("data/raw/landsat_ood")


def drive_service():
    init_ee()
    return build("drive", "v3", credentials=ee.data.get_persistent_credentials())


def list_all(svc, folder_id):
    files, tok = [], None
    while True:
        r = svc.files().list(q=f"'{folder_id}' in parents and trashed=false",
                             fields="nextPageToken, files(id,name,size)",
                             pageSize=1000, pageToken=tok).execute()
        files += r.get("files", [])
        tok = r.get("nextPageToken")
        if not tok:
            break
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    svc = drive_service()
    r = svc.files().list(
        q=f"name='{FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)").execute()
    folders = r.get("files", [])
    assert folders, f"folder {FOLDER} not found in Drive"
    files = list_all(svc, folders[0]["id"])
    print(f"folder '{FOLDER}': {len(files)} files in Drive", flush=True)

    have = {p.name for p in OUT.glob("*.tif")}
    todo = [f for f in files if f["name"] not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"already have {len(have)}, downloading {len(todo)}", flush=True)

    n = 0
    for f in todo:
        dest = OUT / f["name"]
        part = dest.with_suffix(dest.suffix + ".part")
        req = svc.files().get_media(fileId=f["id"])
        with io.FileIO(part, "wb") as buf:
            dl = MediaIoBaseDownload(buf, req, chunksize=16 * 1024 * 1024)
            done = False
            tries = 0
            while not done:
                try:
                    _, done = dl.next_chunk()
                except Exception as e:
                    tries += 1
                    if tries > 8:
                        raise
                    print(f"  retry {f['name']} ({str(e)[:40]})", flush=True); time.sleep(5)
        part.rename(dest)
        n += 1
        if n % 100 == 0:
            print(f"  downloaded {n}/{len(todo)}", flush=True)
    print(f"done: downloaded {n} tiles -> {OUT} (total now {len(have)+n})", flush=True)


if __name__ == "__main__":
    main()
