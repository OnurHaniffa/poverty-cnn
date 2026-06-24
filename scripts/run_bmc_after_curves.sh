#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/poverty-cnn"
EPY="$HOME/miniconda3/envs/poverty-cnn/bin/python"
echo "=== waiting for scaling/learning curves to finish ==="
while pgrep -f 33_scaling_and_curves >/dev/null; do sleep 120; done
echo "=== curves done — running BMC mitigation (5-fold) ==="
"$EPY" scripts/37_train_bmc.py --folds A,B,C,D,E
echo DONE > "$HOME/bmc.done"
