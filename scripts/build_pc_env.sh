#!/usr/bin/env bash
# Build the poverty-cnn TRAINING environment on the lab GPU server (Linux/CUDA).
#
# Why pip-for-packages instead of `conda env create -f environment.yml`:
# the PC's connection to package CDNs is ~200 KB/s, and torch (~2.4 GB) dominates
# the build either way. pip pulls the CUDA torch wheel directly (one download),
# whereas conda would install CPU torch then re-download the CUDA build. pip is
# also more resilient to slow/flaky links. Functionally the same env as the Mac's
# conda env; torch/torchvision pinned to match the Mac (2.2.2 / 0.17.2).
#
# Run unattended:  nohup bash ~/build_pc_env.sh > ~/env_build.log 2>&1 &
set -euo pipefail

CONDA="$HOME/miniconda3/bin/conda"
EPY="$HOME/miniconda3/envs/poverty-cnn/bin"

echo "=== [1/5] ensure conda env (python 3.11 + pip, conda-forge) ==="
# conda-forge with --override-channels avoids Anaconda's `defaults` channel
# (which recent Miniconda gates behind a ToS prompt). Idempotent + re-runnable:
# create with pip if missing, else just ensure pip is present.
if "$CONDA" env list | grep -q 'envs/poverty-cnn'; then
    echo "env already exists — ensuring pip is installed"
    "$CONDA" install -y -n poverty-cnn -c conda-forge --override-channels pip
else
    "$CONDA" create -y -n poverty-cnn python=3.11 pip -c conda-forge --override-channels
fi

echo "=== [2/5] pip upgrade ==="
"$EPY/pip" install --no-input --upgrade pip

echo "=== [3/5] torch + torchvision (CUDA 12.1, pinned to match the Mac) ==="
"$EPY/pip" install --no-input torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/5] geospatial + ML + project deps (PyPI) ==="
"$EPY/pip" install --no-input \
    rasterio geopandas shapely pyproj \
    pandas "numpy<2" scipy scikit-learn \
    matplotlib seaborn optuna tensorboard tqdm pyyaml rich \
    earthengine-api wilds grad-cam python-dotenv pytest

echo "=== [5/5] editable install of poverty_cnn ==="
cd "$HOME/poverty-cnn" && "$EPY/pip" install -e .

echo "=== verify ==="
"$EPY/python" - <<'PY'
import torch, rasterio, geopandas, sklearn, ee
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("rasterio/geopandas/sklearn/ee imports OK")
PY
echo "ENV BUILD DONE"
