#!/usr/bin/env bash
# Run the full 23-country bulk extraction on the lab GPU server.
# Resumable: skips countries already in results/extraction_manifest_<CC>.csv,
# and skips individual clusters already submitted. Year is read from each GPS
# file's DHSYEAR field (no caller-supplied year map — that's what broke before).
#
# Run unattended:  nohup bash ~/poverty-cnn/scripts/extract_all_pc.sh > ~/extraction.log 2>&1 &
set -uo pipefail
EPY="$HOME/miniconda3/envs/poverty-cnn/bin/python"
cd "$HOME/poverty-cnn"

for cc in AO BF BJ CD CI CM ET GH GN KE LS ML MW MZ NG RW SL SN TG TZ UG ZM ZW; do
  shp=$(ls data/raw/dhs/$cc/*GE*/*.shp 2>/dev/null | head -1)
  if [ -z "$shp" ]; then echo "!! SKIP $cc: no GPS shapefile"; continue; fi
  echo "===== $cc : $shp ====="
  "$EPY" scripts/03_download_imagery.py --country "$cc" --gps "$shp" || echo "!! $cc errored, continuing"
done
echo "===== PC: ALL 23 COUNTRIES SUBMITTED ====="
