#!/usr/bin/env bash
# Self-healing multi-round "full" pipeline: rclone (retry until complete) ->
# build full cache (pilot + multiround) -> train full 5-fold. Owns the rclone so a
# silent death just resumes (rclone copy is incremental). Gates each step on success.
# Launch detached on the PC:
#   setsid nohup bash scripts/run_full_pipeline.sh > ~/full_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/poverty-cnn"
EPY="$HOME/miniconda3/envs/poverty-cnn/bin/python"
RC="$HOME/.local/bin/rclone"

echo "=== rclone download (retry-until-complete) ==="
tries=0
until "$RC" copy gdrive:poverty_cnn_multiround data/raw/landsat \
      --transfers 32 --checkers 32 --retries 10 --low-level-retries 20 \
      --stats 60s --stats-one-line -v --log-file "$HOME/rclone_multiround.log"; do
  tries=$((tries+1)); echo "rclone exited nonzero (try $tries) — resuming in 60s"; sleep 60
  if [ "$tries" -ge 20 ]; then echo "!! rclone failed 20x"; echo RCLONE_FAILED > "$HOME/full_pipeline.done"; exit 1; fi
done
N=$(ls data/raw/landsat/*.tif 2>/dev/null | wc -l)
echo "=== RCLONE COMPLETE (exit 0): $N tiles in landsat/ ==="
if [ "$N" -lt 20000 ]; then echo "!! only $N tiles (<20000) — partial, NOT proceeding"; echo "LOW_$N" > "$HOME/full_pipeline.done"; exit 1; fi

echo "=== building FULL cache (multiround wealth index + all tiles) ==="
"$EPY" scripts/08_build_tile_cache.py --raw-dir data/raw/landsat \
  --wealth data/processed/multiround_wealth_index_clusters.csv \
  --out data/processed/tile_cache_full --force
[ -f data/processed/tile_cache_full/cache.npy ] || { echo CACHE_FAILED > "$HOME/full_pipeline.done"; exit 1; }

echo "=== training FULL 5-fold (stabilized config) ==="
"$EPY" -m poverty_cnn.training.train --folds A,B,C,D,E \
  --cache data/processed/tile_cache_full --out results/cnn_full
echo DONE > "$HOME/full_pipeline.done"
echo "=== FULL PIPELINE COMPLETE ==="
