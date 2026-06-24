# Poverty CNN

Predicting village-level asset wealth across 23 sub-Saharan African countries from 8-band satellite imagery — a from-scratch replication of Yeh et al. (2020), and an audit of where it breaks.

> Onur Haniffa · ML/DL Internship 2026 · Advisor: Dr. Seda Nilgün Dumlu, Acıbadem MAA University

## The short version

I rebuilt Yeh et al. (2020) in PyTorch: an 8-channel ResNet-18 that reads Landsat surface reflectance plus night-lights and predicts a village's DHS asset-wealth index. Then I spent most of the project on the question the original paper doesn't ask — not *does it work*, but **who does it work for**.

It works, at benchmark level, on countries it never trained on. And it systematically fails the poorest, in a way that no standard fix repairs and that the model's own uncertainty can't even detect. The point of the project isn't a better model — it's mapping exactly where and why satellite poverty estimation hits a wall. That wall is physical: the strongest wealth cue from space is night-lights, and the poorest villages are dark.

## What I found

Every number below is **leave-country-out** — measured on countries held out of training entirely, with 95% confidence intervals from a country-level bootstrap.

**It replicates, honestly.** Pearson r 0.76 [0.72–0.79], r² 0.57 [0.51–0.62], Spearman 0.72 — in the range of published satellite-poverty models, on a harder split than most use. A random train/test split would have inflated r² to 0.73; the country-blocked number is the honest one, so that's what I report.

**It serves cities better than villages.** The absolute error is about the same in both, but ranking collapses for rural villages (Spearman 0.38 vs 0.54 urban) — and rural is the 8,473-village majority that aid targeting actually has to reach.

**It's confidently wrong about the poorest.** Sort villages by true wealth and the model over-predicts the poorest decile by +0.62 index points while under-predicting the richest — a clean regression-to-the-mean staircase, slope 0.60. The poorest come out looking richer than they are, lifted off the danger line on paper.

**Its uncertainty can't catch this** — the part I think is genuinely new. I tried three uncertainty methods (deep ensembles, MC-dropout, a heteroscedastic head). None flag the poor; they're most overconfident exactly where the model is most wrong. The reason is clean: error = noise + bias + variance, uncertainty estimates *variance*, and the poorest's error is *bias* — every model agrees on the same wrong answer, so it reads as confident. Calibration is not equity.

**And no fix repairs it.** Loss reweighting barely moves the bias; a balance-forcing loss makes it worse and wrecks accuracy; tripling the training data lifts the average but leaves the poorest untouched. You can rescale the *level* with one line of arithmetic, but you can't recover *ranking* among the poor — because the information isn't in the pixels. The night-light–wealth correlation is ~0.76 overall but drops to 0.28 within the poorest 30%.

**Robustness.** Ranking transfers to six brand-new countries outside the 23 (Spearman 0.48–0.81), though absolute levels break on the wealthy tail. And the fairness findings survive re-indexing wealth *within* each country, so they aren't an artifact of pooling the wealth index across countries.

### Results at a glance

| metric (leave-country-out) | value |
|---|---|
| Pearson r | 0.76 [0.72–0.79] |
| r² | 0.57 [0.51–0.62] |
| Spearman ρ | 0.72 [0.67–0.76] |
| poorest-decile bias | +0.62 [+0.60–0.64] |
| urban / rural Spearman | 0.54 / 0.38 |
| targeting recall (poorest 20%) | 49% |

The full talk that walks through all of this lives in [`deck/`](deck/) (35 slides, built from HTML via headless Chrome).

## What this replicates and extends

- **Yeh et al.** *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa.* Nature Communications 11:2583, 2020. [doi:10.1038/s41467-020-16185-w](https://doi.org/10.1038/s41467-020-16185-w) — the replication target (their code is in unmaintained TF 1.15).
- **Jean et al.** *Combining satellite imagery and machine learning to predict poverty.* Science 353(6301):790–794, 2016. [doi:10.1126/science.aaf7894](https://doi.org/10.1126/science.aaf7894) — the transfer-learning baseline.
- **Aiken, Rolf & Blumenstock.** *Fairness and representation in satellite-based poverty maps.* IJCAI 2023. [arXiv:2305.01783](https://arxiv.org/abs/2305.01783) — the urban–rural fairness audit, which I extend from their 10 countries to all 23.

## Quickstart

```bash
conda env create -f environment.yml
conda activate poverty-cnn
earthengine authenticate          # free Google Earth Engine account, ~1 day approval
```

DHS asset-survey data needs a free registration (1–3 day approval) at [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm) — one application covers all 23 countries. The micro-data is restricted and **never** committed here; the pipeline expects it under `data/raw/` (gitignored).

```python
from poverty_cnn.data.dhs import extract_asset_features, pooled_wealth_index
from poverty_cnn.data.dataset import PovertyTileDataset, make_fold_loaders
from poverty_cnn.data.splits import fold_ids                  # 5-fold leave-country-out CV
from poverty_cnn.models.poverty_resnet import PovertyResNet   # 8-channel ResNet-18
from poverty_cnn.training.train import train_fold             # Adam + MSE, early stop on val r²
```

Stage scripts live in `scripts/`, numbered by pipeline order (wealth index → imagery → tile cache → train → evaluate → audit).

## Data sources

| Source | Access | License |
|---|---|---|
| DHS asset surveys (23 countries) | [dhsprogram.com](https://dhsprogram.com) | free, registration required |
| Landsat 5/7/8 surface reflectance | Google Earth Engine | US public domain |
| DMSP-OLS / VIIRS night-lights | Earth Engine, NOAA | public domain |
| WILDS PovertyMap (sanity check) | `wilds` package | MIT |

## Project structure

```
poverty-cnn/
├── src/poverty_cnn/      # importable package: data, models, training, eval, viz
├── scripts/              # numbered entry-point scripts (one per pipeline stage)
├── deck/                 # the presentation (HTML build system + rendered PDF)
├── docs/                 # methodology-defense.md, results.md, design notes
├── tests/                # pytest
├── data/                 # gitignored — raw + processed DHS/imagery
└── results/              # model outputs, metrics, figures
```

## Reproducibility

Seeds are fixed and logged, the conda environment is pinned in `environment.yml`, and every result above is regenerable from the numbered `scripts/`. The honest-evaluation choices — leave-country-out splits, country-level bootstrap CIs, within-country robustness checks — are the point, so they're all in code rather than asserted.

## License

MIT (`LICENSE`). If you build on this, please also cite the original papers (Jean 2016, Yeh 2020, Aiken 2023).
