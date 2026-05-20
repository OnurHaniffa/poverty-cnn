# Poverty CNN

A modernized replication and fairness audit of Yeh et al. (2020), *Nature Communications*: predicting village-level asset wealth across 23 African countries from publicly-available Landsat satellite imagery.

> Onur Haniffa · ML/DL Internship, Spring 2026 · Advisor: Dr. Seda Nilgün Dumlu

## What this is

This project does four things:

1. **Replicates** Yeh et al. 2020's headline result (mean cross-country r² = 0.70) on a clean PyTorch 2.x reimplementation of their TensorFlow 1.15 codebase.
2. **Extends** Aiken, Rolf, Blumenstock 2023 (*IJCAI*) urban-rural fairness audit from 10 countries to all 23.
3. **Novel — adds uncertainty-aware fairness analysis.** Uses MC-dropout to estimate per-cluster prediction uncertainty, shows that uncertainty itself is unequally distributed across urban/rural strata, and proposes an uncertainty-aware aid allocation rule.
4. **Novel — adds temporal fairness drift analysis.** Trains on early DHS surveys (Period 1+2: 2009–2014), evaluates on later (Period 3: 2015–2017+), and shows how the urban-rural fairness gap evolves over time.

Design and methodology document: [`docs/design.md`](docs/design.md) (mirror of the spec at `ivory-ai/docs/superpowers/specs/2026-05-08-poverty-cnn-internship-design.md`).

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

The project is under active development. As of this commit, the importable package
exposes the data-loading half of the pipeline:

```python
from poverty_cnn.data.dhs import (
    load_dhs_hr,
    load_dhs_pr_as_hr,
    load_dhs_gps,
    extract_asset_features,
    pooled_wealth_index,
)
from poverty_cnn.data.earth_engine import (
    init_ee,
    cluster_image,
    download_cluster_tile_direct,
    export_cluster_to_drive,
)
```

CLI entry-points (`scripts/0X_*.py`) for the full end-to-end pipeline —
imagery download, dataset build, training, evaluation, fairness audit — will
land as the corresponding modules under `src/poverty_cnn/` are completed.
See [`docs/tasks.md`](docs/tasks.md) for current progress and
[`docs/design.md`](docs/design.md) §11 for the planned timeline.

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
