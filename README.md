# Poverty CNN

A modernized replication and fairness audit of Yeh et al. (2020), *Nature Communications*: predicting village-level asset wealth across 23 African countries from publicly-available Landsat satellite imagery.

> Onur Haniffa · ML/DL Internship, Spring 2026 · Advisor: Dr. Seda Nilgün Dumlu

## What this is

A clean PyTorch 2.x reimplementation of Yeh et al. (2020), whose original code is in the now-unmaintained TensorFlow 1.15. There are four goals, in roughly this order:

1. **Replicate** the paper's headline result — mean cross-country r² ≈ 0.70 — from scratch.
2. **Extend** the Aiken, Rolf & Blumenstock (2023, *IJCAI*) urban–rural fairness audit from their 10 countries to all 23.
3. **Add an uncertainty angle** (novel): use MC-dropout to estimate per-cluster prediction uncertainty, then check whether that uncertainty is itself unevenly spread across urban/rural strata — and if so, what an uncertainty-aware aid-allocation rule would do about it.
4. **Add a temporal angle** (novel): train on the earlier DHS surveys (2009–2014), test on the later ones (2015–2017), and see whether the urban–rural gap widens over time.

Status: the replication pipeline (data → CNN → cross-country CV) is the current focus. Goals 3 and 4 are planned extensions, not yet run — so don't read the bullets above as finished results. [`docs/tasks.md`](docs/tasks.md) tracks where things actually stand; [`docs/design.md`](docs/design.md) has the full methodology.

## Replicated papers

- Jean, Burke, Xie, Davis, Lobell, Ermon. *Combining satellite imagery and machine learning to predict poverty.* Science 353(6301):790–794, 2016. DOI: [10.1126/science.aaf7894](https://doi.org/10.1126/science.aaf7894).
- Yeh, Perez, Driscoll, Azzari, Tang, Lobell, Ermon, Burke. *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa.* Nature Communications 11:2583, 2020. DOI: [10.1038/s41467-020-16185-w](https://doi.org/10.1038/s41467-020-16185-w).

## Critique extended

- Aiken, Rolf, Blumenstock. *Fairness and representation in satellite-based poverty maps: Evidence of urban-rural disparities and their impacts on downstream policy.* IJCAI 2023. arXiv: [2305.01783](https://arxiv.org/abs/2305.01783).

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate poverty-cnn
```

### 2. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Requires a Google Earth Engine account (free, ~1 day approval): [earthengine.google.com](https://earthengine.google.com).

### 3. Register for DHS data

DHS asset survey data requires registration (free, 1–3 day approval): [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm). Single application covers all 23 sub-Saharan African countries used in this project.

### 4. Use the package

The package now covers the data and modeling pieces end to end:

```python
from poverty_cnn.data.dhs import extract_asset_features, pooled_wealth_index
from poverty_cnn.data.dataset import PovertyTileDataset, make_fold_loaders  # (image, wealth) pairs
from poverty_cnn.data.splits import fold_ids                                # 5-fold cross-country CV
from poverty_cnn.models.poverty_resnet import PovertyResNet                 # 8-channel ResNet-18
from poverty_cnn.training.train import train_fold                           # Adam + MSE, early stopping
```

Stage scripts live in `scripts/`, numbered by pipeline order (wealth index → imagery →
tile cache → train → evaluate → hparam search). The `eval/` modules (fairness, uncertainty,
temporal drift, targeting) are still being filled in — see [`docs/tasks.md`](docs/tasks.md)
for current progress.

## Data sources

| Source | Access | License |
|---|---|---|
| DHS asset surveys (23 countries) | [dhsprogram.com](https://dhsprogram.com) | Free, registration required |
| Landsat 5/7/8 surface reflectance | Google Earth Engine | US Public Domain |
| DMSP-OLS / VIIRS nighttime lights | Google Earth Engine, NOAA | Public domain |
| WILDS PovertyMap (sanity-check) | `wilds` Python package | MIT |

## Project structure

```
poverty-cnn/
├── README.md                 # this file
├── environment.yml           # conda env spec
├── pyproject.toml            # Python project metadata
├── docs/
│   ├── design.md             # full design doc
│   └── tasks.md              # progress tracking
├── src/poverty_cnn/          # importable package
│   ├── data/                 # DHS, Earth Engine, Dataset, splits
│   ├── models/               # ResNet-18 + Jean transfer baseline
│   ├── training/             # train loop, hparam search
│   ├── eval/                 # metrics, fairness, uncertainty, temporal, targeting
│   └── viz/                  # plots and maps
├── scripts/                  # entry-point scripts (numbered by stage)
├── notebooks/                # exploration
├── tests/                    # pytest tests
├── data/                     # gitignored: raw + processed data
└── results/                  # gitignored: checkpoints, predictions, figures
```

## Reproducibility

This project follows the NeurIPS reproducibility checklist:
- All random seeds fixed and logged
- Conda environment locked in `environment.yml`
- All hyperparameters logged via TensorBoard
- Single-command reproduction from raw data
- Hardware specifications documented

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the original papers (Jean 2016, Yeh 2020, Aiken 2023).
