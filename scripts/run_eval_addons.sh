#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/poverty-cnn"
EPY="$HOME/miniconda3/envs/poverty-cnn/bin/python"
echo "=== waiting for temporal run to finish ==="
while pgrep -f 31_train_temporal >/dev/null; do sleep 120; done
echo "=== temporal done — running data-scaling + learning curves ==="
"$EPY" scripts/33_scaling_and_curves.py
echo DONE > "$HOME/eval_addons.done"
