# Predicting Asset Wealth from Satellite Imagery

**A modernized replication and fairness audit of Yeh et al. (2020), benchmarked against Jean et al. (2016).**

---

| | |
|---|---|
| Author | Onur Haniffa |
| Advisor | Dr. Seda Nilgün Dumlu |
| Course | ML/DL Internship, Spring 2026 |
| Project window | 8 May – 30 May 2026 (≈ 3 weeks) |
| Primary venues replicated | *Nature Communications* (Yeh 2020), *Science* (Jean 2016) |
| Primary critique extended | *IJCAI 2023* (Aiken, Rolf, Blumenstock) |
| Status | Design draft, awaiting approval |

---

## 0. Executive summary

This project replicates Yeh et al. 2020 (*Nature Communications*) — a deep-learning system that predicts village-level asset wealth across 23 African countries from publicly-available Landsat satellite imagery — and extends it with **a multi-axis fairness audit, an uncertainty-aware extension, and a temporal-drift extension** that go beyond what the original or any published follow-up has done.

The core narrative arc, four contributions:

1. **Replicate** Yeh 2020's headline result (mean cross-country r² = 0.70) using their exact 5-fold cross-country protocol, modernizing the dead TensorFlow 1.15 codebase to **PyTorch 2.x** and current geospatial libraries.
2. **Audit** the model's per-country and urban/rural fairness across all 23 countries, extending Aiken et al. 2023 (*IJCAI*), who audited only 10. The hypothesis from prior work: aggregate r² hides systematic urban-rural disparities. Translate the disparity into a downstream policy claim by stratifying Yeh 2020's targeting-accuracy analysis (Fig. 5b) by urban / rural.
3. **Extend with uncertainty (novel)**. Use MC-dropout to estimate per-cluster prediction uncertainty. Show that uncertainty itself is unequally distributed across urban / rural and across countries. Propose an **uncertainty-aware aid allocation rule** that abstains on high-uncertainty clusters and quantify the policy improvement.
4. **Extend over time (novel)**. Train on early DHS surveys (Period 1+2: 2009–2014), evaluate on later (Period 3: 2015–2017). Show whether the urban-rural fairness gap *widens* or *narrows* over time as African landscapes urbanize. Decompose the temporal drift into distributional-shift components.

Two supporting analyses (calibration via bootstrap CIs and Grad-CAM interpretability) round out the project. A bias-amplification simulation is a stretch goal for the final week if time permits. WILDS PovertyMap is a documented fallback if the raw Earth Engine pipeline collapses.

The project produces one defensible CV sentence:

> *Replicated Yeh et al. 2020 (Nature Communications) cross-country r² = 0.70 on a modern PyTorch 2.x reimplementation of the 23-country DHS + Landsat benchmark, then extended Aiken et al. 2023 (IJCAI) with three novel contributions: an uncertainty-aware fairness analysis exposing that prediction uncertainty itself is urban-rural-disparate, a temporal-drift analysis showing the fairness gap [widens/narrows] over time, and a stratified targeting-accuracy table demonstrating a [X]-point gap in aid-targeting accuracy between urban and rural clusters with implications for satellite-based aid allocation.*

---

## 1. Article summaries

### 1.1 Jean et al. (2016) — *Science*

**Citation.** Jean, Burke, Xie, Davis, Lobell, Ermon. *Combining satellite imagery and machine learning to predict poverty.* Science 353(6301):790–794, 19 August 2016. DOI: 10.1126/science.aaf7894.

**Problem.** Reliable poverty data is scarce in much of the developing world. The authors ask whether high-resolution daytime satellite imagery, combined with publicly available nighttime lights, can predict cluster-level economic outcomes when survey data is sparse.

**Countries and data.** Nigeria, Tanzania, Uganda, Malawi, Rwanda. DHS asset surveys plus LSMS consumption surveys. Daytime imagery: Google Static Maps API (high-resolution RGB). Nighttime lights: DMSP-OLS time series.

**Method (multi-step transfer learning).**
1. Start with a CNN pre-trained on ImageNet (1000-class classification).
2. Fine-tune the CNN to predict cluster-level mean nighttime light intensity from daytime satellite imagery. Nighttime light is a noisy but globally available proxy for economic activity, treated as a 3-class classification problem (low/medium/high).
3. Treat the fine-tuned CNN as a feature extractor: pass each daytime image through it and take an intermediate-layer activation as a fixed feature vector.
4. Reduce features to **100 dimensions via PCA**.
5. Train **ridge regression** on those 100-D features to predict average household consumption or cluster-level asset wealth.

**Headline numbers (verbatim from Figs. 3, 4, 5 of the paper).**
- Asset wealth, country-specific cross-validated r² across 5 countries: **55–75%**.
- Consumption, country-specific cross-validated r²: Nigeria 2012 = 0.42, Tanzania 2012 = 0.55, Uganda 2011 = 0.41, Malawi 2013 = 0.37.
- Pooled (all 4 LSMS countries) consumption r²: 44–59%.
- Cross-border (train one country, evaluate on another): r² generally lower but still meaningful; pooled-trained models perform almost as well within-country as country-specific models.

**Stated limitations.**
- Nighttime lights are an imperfect proxy and saturate at low expenditure levels.
- Daytime images are not date-stamped to the survey, introducing temporal mismatch.
- DHS GPS coordinates are jittered up to 10 km in rural areas for privacy, adding locational noise.
- Performance is weaker for distinguishing variation among the very poor.

### 1.2 Yeh et al. (2020) — *Nature Communications*

**Citation.** Yeh, Perez, Driscoll, Azzari, Tang, Lobell, Ermon, Burke. *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa.* Nature Communications 11:2583, 22 May 2020. DOI: 10.1038/s41467-020-16185-w. Code & data: `github.com/sustainlab-group/africa_poverty`.

**Problem and scope.** A direct end-to-end successor to Jean 2016. Predicts village-level asset wealth across **19,669 villages in 23 African countries** from DHS surveys conducted between 2009 and 2016 (>500 k households).

**Target.** **Asset wealth index**, defined as the first principal component of household responses to DHS asset-ownership questions (rooms occupied, electricity, floor / wall / roof quality, water source, toilet type, ownership of phone / radio / TV / motorbike), plus 1–5 quality scores. Cluster (village) average is the regression target. Standardized to mean 0, std 1 across all households across all 23 countries.

**Imagery (8 channels, 224 × 224, ≈ 6.72 km on a side).**
- 7 multispectral bands from Landsat 5/7/8 surface reflectance via Google Earth Engine: **RED, GREEN, BLUE, NIR, SWIR1, SWIR2, TEMP1**.
- 1 nightlight band: **DMSP-OLS** for 2009–2011, **VIIRS** for 2012–2017.
- Three-year median composites used to reduce cloud / short-term noise. Composites: 2009–11, 2012–14, 2015–17.

**Architecture.** ResNet-18 (v2 with pre-activation), pre-trained on ImageNet. Modifications:
- First convolutional layer is replaced to accept 8 input channels. Pre-trained ImageNet weights are reused for the 3 RGB channels (scaled by 3/C where C is total input channels). Remaining 5 channels are initialized using truncated-normal random initialization scaled to match ImageNet RGB statistics.
- Final fully-connected layer is replaced by a **single linear unit (scalar regression)**.
- BatchNorm after every convolutional layer (no learned biases).

**Training.**
- Optimizer: **Adam** with default β₁ = 0.9, β₂ = 0.999.
- Loss: **MSE**.
- Batch size: **64**.
- Learning rate: decayed by **0.96 each epoch** from initial value.
- Epochs: **150 for in-country, 200 for out-of-country**.
- Augmentation: random horizontal/vertical flips, brightness adjustment ± 0.5 σ, contrast ± 0.5, randomized order of stacked bands.
- Hyperparameter grid: lr ∈ {1e-2, 1e-3, 1e-4, 1e-5}, L2 ∈ {1e-0, 1e-1, 1e-2, 1e-3}.
- Final fully-connected layer is fine-tuned via **leave-one-group-out cross-validation** for regularization (an extra step on top of the 5-fold over countries).

**Cross-validation scheme.**
- **5-fold over countries**: 23 countries are manually partitioned into 5 folds of roughly equal village count (assignment in Supplementary Table S2). Each rotation trains on 4 folds (~18 countries), tests on the 5th (~5 countries). Headline result is the mean across 5 rotations.
- **In-country setting**: within each country, 5-fold over villages with DBSCAN spatial grouping to prevent overlap between train and test.

**Headline results (from Figs. 2, 3 of the paper).**

| Model | r² pooled | r² mean (country avg.) |
|---|---|---|
| **CNN MS+NL** (multispectral + nightlights, end-to-end) | **0.67** | **0.70** |
| CNN NL (nightlights only) | 0.66 | 0.70 |
| KNN scalar NL | 0.66 | 0.69 |
| CNN MS (multispectral only) | 0.62 | 0.65 |
| CNN transfer (Jean 2016 method) | 0.56 | 0.60 |
| Linear scalar NL | 0.15 | 0.42 |

- Performance for individual held-out countries is **never below 0.50** and exceeds 0.80 in some; median ≈ 0.704.
- **District-level aggregation**: r² up to **0.83** weighted, **0.72** unweighted.
- **Independent census validation** (8 countries, district level): r² up to **0.89** weighted.
- **Urban / rural held-out (Fig. 3d)**: urban r² = **0.40**, rural r² = **0.32**. The aggregate metric hides this gap.
- **Targeting accuracy** at 30th-percentile threshold (Fig. 5b): MS+NL ≈ **81%**, transfer learning ≈ 75%, scalar NL ≈ 62%.
- **Predicting changes over time**: r² ≈ 0.15 with multispectral, < 0.01 with nightlights only.
- **GPS jitter effect**: adding extra noise reduces r² by ≈ 0.07.

**Stated limitations.**
- Aggregate r² masks performance loss in heterogeneous local environments.
- Locational noise from DHS GPS jitter reduces r² by ≈ 0.07 globally; uneven across urban (2 km jitter) vs rural (10 km jitter).
- Performance is worse at separating the very poor from the near-poor.
- Performance over time is much weaker than performance over space.
- The CNN is a black box; features it relies on are not directly interpretable for policy use.

**Computational footprint reported by the authors.** Training the final Nigeria pixel-level wealth map took **≈ 4 h on a single NVIDIA Titan X GPU**, plus ≈ 24 h of imagery processing.

---

## 2. Related work and the literature gap

The state of the art has moved on since Yeh 2020. Six follow-up papers structure my critique angles:

| Paper | Year | Venue | Core finding | What it leaves open |
|---|---|---|---|---|
| **Aiken, Rolf, Blumenstock** — *Fairness and representation in satellite-based poverty maps* (arXiv:2305.01783) | 2023 | IJCAI | Urban-rural fairness gaps in 10 countries; downstream policy harm to rural populations | **Only 10 countries, not the full 23**; no targeting-accuracy stratification |
| **Sarmadi et al.** — *Towards explaining satellite based poverty predictions with CNNs* (arXiv:2312.00416) | 2023 | IEEE SoutheastCon | Grad-CAM analysis of what features drive the prediction | Single dataset, no cross-country interpretability comparison |
| **Tucker et al.** — *Human bias and CNNs' superior insights in satellite based poverty mapping* | 2024 | *Scientific Reports* (Nature) | CNNs outperform humans but make different kinds of errors | Only one country, no fairness lens |
| **Hall** — *Review of ML and satellite imagery for poverty prediction* | 2023 | *J. International Development* | Review of ~50 papers; no standardized uncertainty benchmark; limited downstream policy translation | Identifies gap but does not fill it |
| **Wang et al.** — *Mitigating Urban-Rural Disparities in Contrastive Representation Learning* (arXiv:2211.08672) | 2022 | — | Self-supervised pretraining narrows the urban-rural gap | Suggests a fix but predates a solid measurement of the gap across all 23 countries |
| **Tempov foundation model** (arXiv:2604.23166) | 2026 | Preprint | Vision Transformer + self-supervised pretraining achieves r² ≈ 0.85–0.87 on Malawi (vs Yeh's 0.70) | Different architecture; doesn't address fairness |

**The gap I sit in.** Aiken 2023 audited 10 countries. Yeh 2020 trained on 23. No public follow-up has audited all 23 with a faithful PyTorch reimplementation. That's the niche.

---

## 3. Project goals and non-goals

### 3.1 Primary goal

Audit Yeh et al. 2020's per-country and urban-rural fairness across all 23 African countries, **and extend the audit with two genuinely novel contributions**: an uncertainty-aware fairness analysis and a temporal-drift analysis. Achieved by (a) a faithful modernized replication of the Yeh 2020 model and 5-fold cross-country protocol, (b) careful per-subgroup analysis on the resulting predictions, (c) MC-dropout uncertainty quantification with stratified aid-targeting simulation, (d) Period-1+2-trained-on, Period-3-tested-on temporal evaluation with urban/rural stratification.

### 3.2 Secondary goals

1. **Modernization contribution**: Port the Yeh 2020 pipeline from TensorFlow 1.15 to PyTorch 2.x with current geospatial libraries (rasterio, earthengine-api). Document any divergence from the original.
2. **Replication of Yeh 2020 headline**: 5-fold cross-country r² target ≈ 0.67 pooled, 0.70 mean (paper's reported numbers).
3. **Direct comparison to Jean 2016 baseline**: Replicate the multi-step transfer learning approach and report its r² alongside the end-to-end model.
4. **Three classical baselines**: linear scalar nightlights regression, KNN on scalar nightlights, ridge regression on handcrafted features (color histograms, NDVI, mean nighttime intensity).
5. **Calibration analysis**: bootstrap 95% CIs on r²; calibration plot of predicted vs actual wealth deciles; expected calibration error (ECE).
6. **Grad-CAM interpretability** on a curated sample of correctly and incorrectly predicted clusters.

### 3.3 Novel contributions (the differentiators)

These are detailed in §7.8 (uncertainty-aware fairness) and §7.9 (temporal drift). In summary:

- **Uncertainty-aware fairness extension**: combine MC-dropout uncertainty with the urban/rural fairness lens. Show that uncertainty itself is unequally distributed. Propose and simulate an uncertainty-aware aid allocation rule. **No published follow-up has done this at the 23-country scale.**
- **Temporal fairness drift**: train on early DHS surveys, evaluate on later ones. Show how the urban-rural fairness gap evolves as African landscapes change over time. Decompose the drift into its components. **No published follow-up has stratified Yeh 2020's temporal r² by urban/rural.**

### 3.4 Non-goals (explicit)

- Predicting changes in wealth over time. Yeh 2020 itself reports r² = 0.15 for time-series — much weaker than spatial. Out of scope.
- Architecture novelty. ResNet-18 is the chosen architecture, matching the paper.
- Pixel-level wealth map generation (the Yeh 2020 24-h imagery pipeline for Nigeria). Out of scope for time.
- Self-supervised pretraining or vision transformer replacement. Future work — too expensive for 3 weeks.
- Building footprints integration. Future work.
- An interactive demo or web app.

---

## 4. Experimental design

### 4.1 The 5-fold cross-country protocol

The protocol that produced Yeh 2020's headline number, replicated exactly:

1. **Fold assignment.** 23 countries are partitioned into 5 disjoint folds (≈ 4–5 countries per fold), balanced by total sample size. **The exact assignment is taken from Yeh 2020 Supplementary Table S2** to enable apples-to-apples comparison.
2. **Rotation.** For each fold *k* ∈ {1, …, 5}: train on folds {1, …, 5} \ {*k*}, test on fold *k*. Five models trained total. Each country is in the test set exactly once and in the training set exactly four times.
3. **Within-fold validation.** Inside each rotation's training data, hold out one of the 4 train folds as a validation split for early stopping and final-FC-layer regularization tuning. (Matches paper.)
4. **Reporting.** Compute r² and Pearson r for each rotation's held-out predictions. Headline number = mean ± std across 5 rotations. Apples-to-apples with Yeh 2020's r² = 0.70 mean.

**Why not 7- or 8-fold.** With 23 countries / 8 folds = ~3 test countries per rotation — high variance per-rotation r². Standard cross-validation theory and the paper's choice both land on 5.

**Why not exhaustive.** "All combinations of 23 choose 5" = 33,649 train/test splits = 33,649 model trainings. Not what k-fold means.

### 4.2 Stretch: leave-one-country-out (LOCO)

If Week 2 finishes ahead of schedule, repeat the experiment as 23-fold (leave-one-country-out). Each country is the sole test set, the other 22 are training. More rigorous per-country r² estimates because every test model saw the maximum possible training data. Cost: ~25–46 extra GPU-hours.

### 4.3 In-country supplementary experiment

For a subset of countries with abundant data (Nigeria, Kenya, Tanzania), run the in-country 5-fold-over-villages protocol with DBSCAN spatial grouping. This gives within-country r² for direct comparison to Jean 2016's per-country numbers (Nigeria 0.42, Tanzania 0.55, etc.). Optional, as time allows.

---

## 5. Data plan

### 5.1 Sources

| Source | Access | Cost | Approx. size |
|---|---|---|---|
| **DHS asset surveys** for all 23 countries (most recent 2009–2018 round per country) | [dhsprogram.com](https://dhsprogram.com), free, single application covers all African countries (1–3 day approval) | Free | < 1 GB CSVs + GPS shapefiles |
| **Landsat 5/7/8 Surface Reflectance** | Google Earth Engine collections: `LANDSAT/LT05/C02/T1_L2`, `LANDSAT/LE07/C02/T1_L2`, `LANDSAT/LC08/C02/T1_L2` | Free, EE account approval (~1 day) | 30–50 GB tiles (~19,000 clusters) |
| **DMSP-OLS Nighttime Lights** (2009–2011 surveys) | Earth Engine `NOAA/DMSP-OLS/NIGHTTIME_LIGHTS` | Free | < 2 GB |
| **VIIRS Nighttime Lights** (2012+ surveys) | Earth Engine `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | Free | < 3 GB |
| **WILDS PovertyMap** (sanity-check companion) | `wilds` Python package, one-line download | Free | ~13 GB |

### 5.2 Asset wealth index construction

Following Yeh 2020 §Methods exactly:

1. From each DHS round, extract household-level responses to the standard asset-ownership questions:
   - Number of rooms occupied (`HV216`)
   - Electricity (`HV206`)
   - Floor / wall / roof material quality scores (1–5 ordinal, derived from `HV213`/`HV214`/`HV215`)
   - Water source quality (1–5 ordinal, derived from `HV201`)
   - Toilet quality (1–5 ordinal, derived from `HV205`)
   - Phone, radio, TV, motorbike (binary: `HV221`, `HV207`, `HV208`, `HV212`)
2. Pool households across **all 23 countries**.
3. Run PCA on the standardized binary/ordinal asset matrix.
4. Take the first principal component as the household-level asset wealth index.
5. Standardize the index to mean 0, std 1 across all households.
6. For each cluster, compute the **average household wealth index**. This is the regression target.

### 5.3 Image preprocessing pipeline

For each DHS cluster *c* with GPS coordinates (lat, lon) in survey year *y*:

1. Determine the matching 3-year composite window: 2009–11 if *y* ∈ {2009,2010,2011}; 2012–14 if *y* ∈ {2012,2013,2014}; 2015–17 if *y* ∈ {2015,2016,2017,2018}.
2. Construct a 6.72 km × 6.72 km bounding box centered on the (jittered) cluster GPS.
3. Query Earth Engine for the per-pixel median across all clear-sky observations in the composite window for the appropriate Landsat collection. Compute the 7 multispectral bands (RED, GREEN, BLUE, NIR, SWIR1, SWIR2, TEMP1).
4. Query the matching nighttime lights collection (DMSP for early years, VIIRS for later) for the same region and window.
5. Resample to 30 m / pixel grid → 224 × 224 pixels nominal, fetch a 255 × 255 patch and centre-crop to 224 × 224 (matches paper).
6. Stack into an 8-channel float32 tensor.
7. Save as `<country>_<cluster_id>_<year>.npy` plus a parallel `metadata.csv` with columns: `country`, `cluster_id`, `lat`, `lon`, `year`, `urban`, `wealth_index`, `n_households`, `region`.

### 5.4 Cross-validation against WILDS PovertyMap

Before training anything, sanity-check the raw pipeline:

1. `pip install wilds` → `dataset = wilds.get_dataset('poverty', download=True)`.
2. Filter to the same countries / years as my pipeline.
3. Compare:
   - Distribution of wealth indices (KS-test, mean, std).
   - Per-band pixel statistics (mean, std).
   - Sample counts per country.
4. If matches within rounding: pipeline is correct. If not: debug before training.

### 5.5 Background download strategy

Earth Engine queries are network-bound. Strategy:
- **Day 1 evening**: launch background download script (10 concurrent EE queries with retries and resumability).
- **Realistic completion**: 24–48 hours of background pull time.
- **Foreground work in parallel**: env setup, code reading, DHS preprocessing.

---

## 6. Model and training plan

### 6.1 Architecture

`PovertyResNet18` (PyTorch 2.x):

```python
class PovertyResNet18(nn.Module):
    def __init__(self, in_channels: int = 8, dropout_p: float = 0.0):
        super().__init__()
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        # Replace first conv: 8 input channels.
        new_conv1 = nn.Conv2d(in_channels, 64,
                              kernel_size=7, stride=2,
                              padding=3, bias=False)
        with torch.no_grad():
            # RGB channels: copy ImageNet weights, scaled by 3/C.
            old_w = backbone.conv1.weight  # (64, 3, 7, 7)
            scale = 3.0 / in_channels
            new_conv1.weight[:, :3] = old_w * scale
            # Non-RGB channels: truncated normal, matching ImageNet std.
            std = old_w.std().item()
            nn.init.trunc_normal_(new_conv1.weight[:, 3:], mean=0.0,
                                  std=std, a=-2*std, b=2*std)
        backbone.conv1 = new_conv1
        # Optional dropout for MC-dropout uncertainty experiments.
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(backbone.fc.in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)  # (B,) scalar regression
```

### 6.2 Loss, optimizer, schedule

| | |
|---|---|
| Loss | MSE |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999) |
| Initial LR | 1e-3 (tuned on grid; see §6.4) |
| LR schedule | Exponential decay, factor 0.96 per epoch |
| Batch size | 64 (fall back to 32 if VRAM-tight) |
| Epochs | 150 (200 for out-of-country, matching paper); early stopping patience 15 on val MSE |
| Weight decay | tuned on grid |
| Random seed | fixed per fold for reproducibility |

### 6.3 Augmentation (matching paper)

- Random horizontal flip (p = 0.5)
- Random vertical flip (p = 0.5)
- Per-pixel brightness jitter ± 0.5 σ on multispectral bands
- Per-pixel contrast jitter ± 0.5 σ
- Randomized stacking order of bands within each input

### 6.4 Hyperparameter search

| Axis | Values | Notes |
|---|---|---|
| Learning rate | 1e-3, 1e-4, 1e-5 | Paper covers wider grid; we trim |
| L2 weight decay | 0, 1e-2, 1e-3 | |
| Dropout p (final FC) | 0.0, 0.2 | For MC-dropout option |
| Loss | MSE, Huber (δ = 1.0) | |
| Frozen layers | none, freeze backbone | |

**Search strategy.** Optuna with TPE (Tree-structured Parzen Estimator), **20 trials**, evaluating on a single train/val split (fold 1's training data, with 20% held out for validation). Top config wins, then run the full 5-fold cross-country with that config.

**Why 20 trials, not 30.** Marginal gain past 20 is small under TPE; 30 was over-budget given the dominant cost is the final 5-fold.

### 6.5 Ablation suite

| Ablation | Setup | # models | GPU-h |
|---|---|---|---|
| **CNN MS+NL** (primary) | 8-channel input, full 5-fold | 5 | 5–10 |
| **CNN MS only** | 7-channel multispectral, fold 1 only | 1 | 1–2 |
| **CNN NL only** | 1-channel nightlights, fold 1 only | 1 | 1–2 |
| **Jean 2016 transfer** | nightlights as proxy → ridge regression, full 5-fold | 5 | 5–10 |
| **Linear scalar NL** | mean nightlight intensity → linear regression | 0 | < 0.1 (CPU) |
| **KNN scalar NL** | mean nightlight intensity → k-NN regression | 0 | < 0.1 (CPU) |
| **Ridge on handcrafted features** | color histograms + NDVI + mean NL → ridge regression | 0 | < 1 (CPU) |

Single-fold ablations for MS-only and NL-only are sufficient to confirm or refute Yeh's Fig. 3a numbers without the full 5× cost.

---

## 7. The fairness audit (PRIMARY contribution)

This is the substantive, headline-grade analysis layered on top of the replication.

### 7.1 Hypothesis

The model's aggregate r² = 0.70 conceals systematic disparities along two axes:
- **Geographic**: per-country r² varies widely; some countries are predicted much better than others.
- **Urban / rural**: rural clusters are predicted with systematically larger errors than urban clusters, with downstream policy consequences for aid allocation that targets rural populations.

### 7.2 Subgroups audited

Each held-out cluster is labeled by:

| Attribute | DHS field | Why it matters |
|---|---|---|
| Country code | DHS file | Per-country breakdown of all 23 countries |
| Urban vs Rural | `HV025` (URBAN_RURA) | Aiken 2023's central axis |
| Wealth quintile | derived from ground-truth wealth index | Are the poorest predicted as well as the middle? |
| Region within country | `HV024` (REGION) | Sub-country geography |
| Female-headed household share *(stretch)* | `HV219` aggregated to cluster | Gendered representation |

### 7.3 Per-subgroup metrics

For each subgroup *g*, compute on the held-out predictions from the 5-fold model:

| Metric | Definition | Why |
|---|---|---|
| **r²(*g*)** | Coefficient of determination on cluster predictions in *g* | Goodness of fit |
| **Mean signed error** | mean(predicted − actual) over *g* | Bias direction; negative = under-prediction, suggesting aid misallocation away from *g* |
| **Mean absolute error (MAE)** | mean(|predicted − actual|) | Typical error magnitude |
| **Calibration error (ECE)** | Expected Calibration Error over 10 deciles of predicted wealth | Are predicted values well-calibrated within *g*? |
| **Pearson r** | Linear correlation | Robust to outliers vs r² |

Output: a 23 × 2 (country × urban/rural) matrix per metric.

### 7.4 Statistical formalization

For each disparity claim:

1. **Bootstrap 95% CIs.** 1,000 resamples of held-out predictions. Per-group r² with CI.
2. **Bootstrap CI on the gap.** E.g., "rural r² minus urban r² = −0.08, 95% CI [−0.12, −0.04]." If CI excludes zero, the gap is statistically significant.
3. **Permutation test.** 10,000 permutations of the urban/rural label, recompute the gap each time, see what fraction of the null is more extreme than the observed gap.
4. **Effect size.** Cohen's d on the residual distributions across groups.

Output: a table of per-country gaps with CI, p-value, effect size, sample size.

### 7.5 Targeting accuracy stratified analysis (the killer figure)

Yeh 2020 Fig. 5b shows targeting accuracy at the aggregate level: at the 30th-percentile threshold, the MS+NL model correctly identifies 81% of poor villages.

I extend this:

1. Pick targeting thresholds *p* ∈ {10, 20, 30, 40, 50}th percentiles.
2. For each threshold:
   - **Ground-truth positive set** = clusters with true wealth index below *p*.
   - **Model-predicted positive set** = clusters with predicted wealth index below *p*.
   - **Targeting accuracy** = |intersection| / |model-predicted positive set| (precision) — matches Yeh.
   - **Recall** = |intersection| / |ground-truth positive set|.
3. Stratify each metric by urban vs rural.

The headline policy claim, in the form Yeh 2020 cannot make:

> *At the 30th-percentile aid-targeting threshold, the model correctly targets [81]% of urban villages but only [X]% of rural villages, a gap of [16] percentage points (95% CI […, …], p < 0.001). If deployed for aid allocation across the 23 countries' rural populations, this gap would systematically misallocate aid away from the populations most aid programs are designed to reach.*

The bracketed numbers are TBD until the model trains. The structure of the sentence is locked.

### 7.6 Why-the-gap diagnostic experiments

Two short experiments to attempt a causal-ish explanation, not just a measurement:

1. **GPS jitter contribution.** DHS jitters rural cluster GPS up to 10 km, urban only 2 km. So rural ground truth is positionally noisier. Test: does the urban/rural gap shrink when we restrict evaluation to clusters whose GPS jitter is bounded? (Implementation: re-evaluate on a low-jitter subset.) Yeh's own experiment (which finds aggregate r² loss of 0.07 from added jitter) is the methodology template.
2. **Visual signal scarcity.** Rural clusters have less built environment for the CNN to detect. Test: does prediction error correlate with cluster building density (from Microsoft Open Buildings or Google Open Buildings, a separate data source we can pull lightly)?

Either or both are valid Week 3 stretches. They turn the audit from descriptive into mechanistic.

### 7.7 Visual artifacts (slide deck gold)

1. **Map of Africa** colored by per-country urban-rural r² gap. One panel that shows where the model fails worst.
2. **Forest plot** of per-country r² with urban and rural side by side, ordered by gap size.
3. **Targeting confusion matrix** — urban vs rural side by side, multiple thresholds.
4. **Calibration plot** — predicted vs actual wealth deciles, one panel per group.
5. **Residual scatter** — predicted minus actual wealth on Y-axis, true wealth on X-axis, colored by urban/rural.

### 7.8 Uncertainty-aware fairness extension (novel)

**Hypothesis.** Beyond the per-subgroup *accuracy* disparities documented in §7.3, the model's *uncertainty* may itself be unequally distributed. If rural clusters have **both** higher prediction error AND higher prediction uncertainty, then naive deployment doubles the harm: the model is confident on (mostly correct) urban predictions and uncertain on (mostly wrong) rural ones, with no mechanism to abstain.

No published follow-up to Yeh 2020 has combined uncertainty quantification with the urban-rural fairness lens at the 23-country scale. This is the project's primary novel contribution.

**Method.**

1. **MC-dropout uncertainty estimation.** Train ResNet-18 with dropout (`p = 0.2`) inserted before the final FC layer (already in our hyperparameter grid in §6.4). At inference time, **keep dropout active**. Run each held-out cluster through the model **50 times** with different dropout masks. Take the mean of the 50 predictions as the point estimate; take the standard deviation as per-cluster uncertainty.
2. **Per-subgroup uncertainty distribution.** For each (country, urban/rural) cell, report:
   - Mean and median per-cluster uncertainty
   - 90th percentile uncertainty (tail risk)
   - Histogram of uncertainties (visualized as violin plot)

   Test the urban-rural uncertainty gap with the same statistical apparatus as §7.4 (bootstrap CI, permutation test).
3. **Uncertainty-aware aid allocation simulation.** For aid targeting at the 30th-percentile threshold:
   - **Naive rule (Yeh 2020 baseline)**: target every cluster predicted below the 30th percentile.
   - **Uncertainty-aware rule (proposed)**: target every cluster predicted below the 30th percentile AND with uncertainty below threshold *t*. Abstain on high-uncertainty clusters (defer to ground-survey verification).
   - Sweep *t* across reasonable values (e.g., 50th, 75th, 90th, 95th percentile of uncertainty distribution). For each *t*, compute:
     - **Coverage**: fraction of clusters not abstained on
     - **Targeting accuracy**: precision among targeted clusters
     - **Stratified targeting accuracy**: separately for urban / rural
   - Plot the trade-off frontier (x-axis: coverage; y-axis: rural targeting accuracy).

**Headline policy claim form (numbers TBD until model trains).**

> *"Abstaining on the [25%] of clusters with highest prediction uncertainty reduces rural mis-targeting by [X]% while excluding only [Y]% of clusters from the program. The remaining clusters can be assigned to ground-survey verification, restoring fairness at a modest operational cost."*

**Time and compute.** Training: zero extra cost (we already train with `dropout = 0.2` as part of the hyperparameter grid). Inference: 50× the per-cluster inference compute, but inference is cheap; ~2 GPU-h total across all 23 countries' held-out predictions. Analysis: ~1.5 days.

### 7.9 Temporal fairness drift (novel)

**Hypothesis.** Yeh 2020 reports that temporal generalization (predicting 2015–2017 from earlier surveys) is much harder than spatial generalization (predicting unseen countries) — cross-time r² ≈ 0.15 vs cross-space r² ≈ 0.70. But Yeh 2020 reports this only at the aggregate level. We hypothesize that temporal drift is *unequally distributed* across urban / rural strata: urban areas urbanize fast (visible in imagery), rural areas change slowly. The fairness gap may therefore *widen* over time.

**Method.**

1. **Temporal split.** Use Yeh 2020's natural composite windows:
   - Period 1: 2009–2011 surveys
   - Period 2: 2012–2014 surveys
   - Period 3: 2015–2017+ surveys

2. **Train on Period 1+2, test on Period 3.** Within each cross-country fold, train only on Period 1+2 clusters; evaluate on Period 3 clusters. This produces a *temporally-held-out* r² in addition to the spatial held-out r² we already have.

3. **Compare against Yeh's reported temporal r².** Yeh 2020 reports r² ≈ 0.15 for temporal generalization. We report our number alongside, with bootstrap CI.

4. **Stratify by urban/rural.** Compute the temporal r² separately for urban and rural Period 3 clusters. Test whether the urban / rural gap widens, stays stable, or narrows over time using the §7.4 statistical apparatus.

5. **Decompose the drift.** Where does the temporal gap come from?
   - **Distributional shift in y**: has the wealth distribution shifted between 2009–2014 and 2015–2017? (KS-test on wealth indices.)
   - **Distributional shift in x**: has the satellite imagery shifted (urbanization, deforestation, road construction)? (Compare per-band pixel statistics across periods.)
   - **Covariate-output relationship shift**: has the mapping from imagery to wealth changed independently of the marginals?

**Headline temporal claim form (numbers TBD).**

> *"Between 2009–2014 and 2015–2017, the urban-rural r² gap [widened / narrowed] from [X] to [Y] (95% CI [...], p < 0.001). Decomposition attributes [Z]% of the temporal degradation to changes in the wealth distribution, [W]% to changes in the imagery, and the remainder to a shift in the covariate-output relationship."*

**Time and compute.** Re-training on Period 1+2 only: 5 additional models (one per fold), ~5–10 GPU-h. Analysis: ~1.5 days.

### 7.10 Time and compute cost (combined fairness audit + extensions)

| Sub-task | Time | GPU |
|---|---|---|
| Per-group metrics + bootstrap CIs (§7.3) | 0.5 d | 0 (pure analysis) |
| Statistical tests (permutation, Cohen's d) (§7.4) | 0.5 d | 0 |
| Targeting accuracy stratified analysis (§7.5) | 0.5 d | 0 |
| Why-the-gap diagnostics (§7.6) | 0.5–1 d | < 2 GPU-h |
| Visualizations (§7.7) | 1 d | 0 |
| Uncertainty-aware extension (§7.8) | 1.5 d | ~ 2 GPU-h |
| Temporal fairness drift (§7.9) | 1.5 d | 5–10 GPU-h |
| **Total fairness block** | **≈ 6 d** | **~10–15 GPU-h** |

Most analysis runs on stored predictions. The temporal extension is the only piece requiring extra training.

---

## 8. Supporting analyses

### 8.1 Calibration via bootstrap CIs

Beyond the per-group calibration in §7.3, two model-level calibration artifacts:

- **Bootstrap distribution of pooled r²** across 1,000 resamples of the held-out predictions. Report the 2.5/50/97.5 percentiles.
- **Calibration plot**: bin predictions into 10 deciles, plot mean predicted vs mean actual within each bin. Yeh 2020 does not do this. ECE is a single-number summary.

Time: 0.5 day. Compute: 0.

### 8.2 Grad-CAM interpretability

Pick 12 clusters from the held-out predictions:
- 4 well-predicted urban (smallest absolute residual)
- 4 well-predicted rural
- 4 worst-predicted (largest absolute residual, mixed urban/rural)

For each, run Grad-CAM on the last convolutional block of the trained ResNet-18. Visualize the attention map overlaid on the RGB version of the satellite tile. Inspect: roads? buildings? vegetation? lights?

Replicates the spirit of Sarmadi et al. 2023 on more diverse geography. Output: a 4 × 3 figure for the slide deck.

Time: 1–2 days. Compute: < 1 GPU-h.

### 8.3 Paired statistical tests on model differences

Where we compare two models on the same held-out predictions (e.g., my MS+NL vs Jean 2016 transfer), use **paired bootstrap test** on r² differences:

1. Resample held-out predictions with replacement (1,000 times).
2. For each resample, recompute r² for both models on the same indices.
3. Distribution of the difference → 95% CI on the gap, with p-value.

This addresses the question "is my model significantly better/worse than baseline X?" rigorously.

Time: 0.5 day. Compute: 0.

---

## 9. Verification stack

Eight defensible answers to "why should anyone believe this works?":

1. **WILDS data parity check** — pre-training. My raw pipeline output statistics (mean, std, distribution) must match the WILDS PovertyMap subset for each country within rounding.
2. **Fold assignment match** — my 5-fold partition of countries matches Yeh 2020 Supplementary Table S2 exactly.
3. **Held-out evaluation only** — every metric is computed on data the model never saw during training (guaranteed by 5-fold protocol).
4. **Bootstrap 95% CIs on every reported metric** — 1,000 resamples; no point estimates.
5. **Compare to published baselines** — Jean 2016 country-specific r² (Nigeria 0.42, Tanzania 0.55, Uganda 0.41, Malawi 0.37); Yeh 2020 r² = 0.67 pooled, 0.70 mean; nightlights-only baseline r² = 0.42 pooled.
6. **Per-country and per-region breakdown** — every country reported separately; 23 numbers, not 1.
7. **Sanity checks** — predictions for known-wealthy areas (Lagos, Abuja, Nairobi, Accra, Cape Town) vs known-poor regions must be directionally correct.
8. **Reproducibility hygiene** — fixed random seeds; full hyperparameter logging via TensorBoard or Weights & Biases; conda env locked in `environment.yml`; reproduction commands in `README.md`.

---

## 10. Computational requirements

### 10.1 Storage

| Item | Size |
|---|---|
| Raw DHS surveys, all 23 countries | < 1 GB |
| Landsat + nighttime tiles for ~19,000 clusters | 30–50 GB |
| Pre-processed PyTorch tensors (8-channel) | 15–25 GB |
| WILDS PovertyMap (sanity-check copy) | ~13 GB |
| Model checkpoints (~10 trained models) | 1–2 GB |
| **Total disk footprint** | **~60–90 GB** — comfortable on a university server |

### 10.2 Compute (estimated)

Anchored on Yeh 2020's reported "≈ 4 h on a Titan X" for one Nigeria-scale model.

| Stage | Estimate |
|---|---|
| One full 150-epoch ResNet-18 run, batch 64, ~14k clusters | 1–2 h on V100/A100; 3–4 h on Titan X |
| Hyperparameter search (Optuna, 20 trials, 1 train/val split) | 25–40 GPU-h |
| Final 5-fold cross-country at best config (with dropout = 0.2 for MC-dropout) | 5–10 GPU-h |
| Ablations (1 fold each for MS-only, NL-only) | 2–4 GPU-h |
| Jean 2016 transfer-learning replication, 5-fold | 5–10 GPU-h |
| Classical baselines (linear NL, KNN NL, ridge) | < 1 GPU-h, mostly CPU |
| Why-the-gap diagnostics + Grad-CAM | < 2 GPU-h |
| **Uncertainty-aware extension (§7.8)** — 50× MC-dropout inference passes | ~ 2 GPU-h |
| **Temporal fairness drift (§7.9)** — 5 retrained models on Period 1+2 only | 5–10 GPU-h |
| **Total estimated GPU budget** | **~50–85 GPU-h** |

This is still below the very-early estimate (95–150 GPU-h) because we trimmed: search to 20 trials, ablations to 1-fold, no architecture comparison. The temporal extension is the only piece adding meaningful new training compute.

### 10.3 Local laptop vs server

| Task | Where |
|---|---|
| Code editing, version control | Laptop, VS Code Remote-SSH into server |
| Data inspection, plotting, reports | Laptop |
| EE downloads | Server background script in `tmux` |
| Model training | Server GPU |
| Analysis on stored predictions | Either |

---

## 11. Three-week timeline

### Week 1 — May 8–15: data and infrastructure

| Day | Work |
|---|---|
| **May 8 (Fri)** | Conda env setup (Python 3.11, PyTorch 2.x, rasterio, earthengine-api). SSH config + VS Code Remote-SSH + `tmux`. Register DHS, request 23-country access. Register Earth Engine. Read this spec end to end with advisor. |
| **May 9 (Sat)** | Read `africa_poverty` repo end-to-end. Read `jmather625/predicting-poverty-replication` end-to-end. Read Aiken 2023 IJCAI paper. **Launch background EE download script** for Landsat + nightlights covering all 23 countries. |
| **May 10 (Sun)** | Pull DHS GPS clusters for all 23 countries (CSV → pandas DataFrame). Compute pooled asset wealth index by PCA. Verify index distribution against WILDS subset. |
| **May 11 (Mon)** | Continue background download. Build PyTorch `Dataset` class for the 8-channel tensors. Write data loading tests. |
| **May 12 (Tue)** | Match Yeh 2020 Supplementary Table S2 fold assignment. Write CV split logic. Start writing `train.py` skeleton. |
| **May 13 (Wed)** | Verify all 23 countries' tiles have downloaded. Verify pixel statistics match WILDS for at least 3 overlap countries. Build modified ResNet-18 in PyTorch. Forward-pass smoke test. |
| **May 14 (Thu)** | Build full training loop with Adam, MSE, lr decay, early stopping, TensorBoard logging. Smoke test: 5 epochs on a 100-cluster subset, confirm loss decreases. |
| **May 15 (Fri)** | **Hard checkpoint.** End-to-end pipeline working: dataset loads, model trains, predictions land in a file. If yes, commit and tag `pipeline-v1`. If no, **fall back to WILDS PovertyMap** for the rest of the project. |

### Week 2 — May 16–22: replication and main result

| Day | Work |
|---|---|
| **May 16 (Sat)** | Hyperparameter search (Optuna, 20 trials, fold 1 train/val split). **Constrain search to include `dropout = 0.2`** so the chosen config is MC-dropout-ready. Runs in `tmux` overnight. |
| **May 17 (Sun)** | Pick best config. Launch full 5-fold cross-country training (5 models × ~1.5 h = ~7.5 h). Runs overnight. |
| **May 18 (Mon)** | Check 5-fold results. Compute pooled r² and mean-of-folds r². Bootstrap CIs. Compare to Yeh 0.67 pooled / 0.70 mean. Save predictions for all held-out clusters. |
| **May 19 (Tue)** | Per-country r² for all 23 countries with bootstrap CIs. **Run MC-dropout inference (50 passes) on all held-out predictions** to get per-cluster uncertainty estimates. Save uncertainty estimates alongside point predictions. |
| **May 20 (Wed)** | Single-fold ablations: CNN MS only, CNN NL only. Compare to Yeh's Fig. 3a numbers. |
| **May 21 (Thu)** | Jean 2016 transfer-learning replication (5-fold). **Kick off temporal training overnight**: 5-fold cross-country, but each model trained on Period 1+2 (2009–2014) clusters only. (~5–10 GPU-h, runs in `tmux`.) |
| **May 22 (Fri)** | **Hard checkpoint.** Replication numbers in hand: pooled r², mean-of-folds r², per-country r², ablations, Jean 2016 baseline, MC-dropout uncertainty estimates, temporal training runs complete — all with CIs. If yes, commit `replication-complete`. If not, drop temporal extension first, then ablations, as needed. |

### Week 3 — May 23–30: fairness audit, novel extensions, supporting analyses, write-up

| Day | Work |
|---|---|
| **May 23 (Sat)** | **Fairness audit start.** Per-subgroup metrics for all (country × urban/rural) cells. Bootstrap CIs on per-group r². |
| **May 24 (Sun)** | Statistical formalization: gap CIs, permutation tests, Cohen's d. **Targeting accuracy stratified analysis** at thresholds {10, 20, 30, 40, 50}th percentile. Construct the killer policy claim. |
| **May 25 (Mon)** | **Uncertainty-aware extension (§7.8).** Per-subgroup uncertainty distribution analysis (mean / median / 90th-percentile uncertainty per country × urban/rural cell). Bootstrap CIs on the uncertainty gap. |
| **May 26 (Tue)** | **Uncertainty-aware aid allocation simulation.** Sweep abstention thresholds, compute coverage vs stratified targeting accuracy, plot the trade-off frontier. Construct the uncertainty policy claim. |
| **May 27 (Wed)** | **Temporal fairness drift (§7.9).** Compute Period 3 r² overall and stratified by urban/rural. Test gap evolution. Decompose drift into y-shift, x-shift, relationship-shift. |
| **May 28 (Thu)** | Visualizations (Africa map, forest plot, targeting confusion matrix, calibration plot). Grad-CAM on 12 curated clusters. Why-the-gap GPS-jitter diagnostic. |
| **May 29 (Fri)** | Final classical baselines + paired bootstrap tests on all model comparisons. **Begin writing the final report.** |
| **May 30 (Sat)** | Finish report. Build slide deck. Clean README. Reproducibility instructions. Commit and tag `final`. |

### Optional Week 3 stretches (only attempt if all prior days finished on schedule)

- **Bias-amplification simulation (the most ambitious stretch).** Iteratively allocate aid via the model, apply a parametric "aid effect" to targeted clusters, retrain, allocate again, repeat for 3 simulated rounds. Track whether the urban-rural fairness gap widens or shrinks across rounds. Sensitivity analysis on the aid-effect magnitude. ~3–4 days, ~10 GPU-h. **Only if Day 28 finishes early.**
- LOCO (23-fold) re-run for the primary model.
- Building-density diagnostic for the why-the-gap analysis (Microsoft Open Buildings or Google Open Buildings).
- In-country supplementary experiment for Nigeria/Kenya/Tanzania to compare against Jean 2016 within-country numbers.

---

## 12. Risk register and fallback ladders

| Risk | Probability | Mitigation |
|---|---|---|
| DHS approval delayed > 3 days | Medium | WILDS PovertyMap fallback. |
| Earth Engine quota / rate limits | Medium | Cache aggressively; download once, reuse. |
| GDAL / `rasterio` install hell | Medium | Use conda-forge, not pip. If still stuck, WILDS only. |
| 8-channel ResNet-18 fails to converge | Low | Fall back to RGB-only first; introduce extra channels incrementally. |
| Server disconnect / job killed | Low | All long jobs in `tmux`; checkpoint every 5 epochs; resume from latest. |
| Hyperparameter search blows budget | Medium | Cap Optuna at 20 trials. Aggressive early stopping (patience 10). |
| Day 15 (May 15) checkpoint missed | — | **Fall back to WILDS PovertyMap.** Same paper, same data, clean PyTorch loader, no GDAL. Project still ships. |
| Day 22 (May 22) checkpoint missed | — | Drop temporal extension first, then bias-amplification stretch, then ablations and Jean 2016 baseline as needed. Ship CNN MS+NL replication + fairness audit + uncertainty extension + supporting analyses. Still has at least one novel contribution. |
| Fairness audit finds no gap | Low | The null finding is itself publishable: replicates Yeh's aggregate but extends with rigorous per-subgroup CIs that Yeh did not provide. Frame as confirmation. |

### Fallback ladder (in priority order if things go wrong)

If time pressure forces cuts, drop in this order — each cut preserves a more-defensible project than the next:

1. **Full plan + bias-amplification stretch**: everything below + the iterative-aid simulation.
2. **Full committed plan**: replication + fairness audit + uncertainty extension + temporal extension + Jean baseline + ablations + supporting analyses (calibration, Grad-CAM) + diagnostics.
3. **Drop bias-amplification stretch**: keep all four committed contributions.
4. **Drop why-the-gap diagnostics** (GPS-jitter, building-density): keep the four committed contributions.
5. **Drop Grad-CAM**: keep replication + fairness audit + uncertainty extension + temporal extension.
6. **Drop temporal extension**: keep replication + fairness audit + uncertainty extension. Still has one novel contribution.
7. **Drop uncertainty extension**: keep replication + fairness audit. Back to the "solid undergraduate" plan.
8. **Drop ablations and Jean 2016 baseline**: only classical baselines remain for comparison.
9. **Switch to WILDS**: skip the raw EE pipeline; use the packaged WILDS PovertyMap dataset; lose the data-engineering CV story but keep the methodology story.

---

## 13. Deliverables

1. **GitHub repository** (public): code, environment lockfile, README with one-command reproduction. License: MIT.
2. **Final report** (markdown rendered to PDF), structured like the diabetes paper:
   - Introduction & motivation
   - Article summaries (Jean 2016, Yeh 2020)
   - Related work and literature gap
   - Methods (pipeline, model, training, CV)
   - Results (replication number with CIs, per-country breakdown, ablations, classical baselines)
   - Fairness audit (subgroup metrics, statistical tests, targeting accuracy)
   - Supporting analyses (calibration, Grad-CAM)
   - Discussion (gap mechanisms, policy implications)
   - Limitations & future work
   - References
3. **Slide deck** (Google Slides, mirroring the diabetes deck) for the internship presentation.
4. **Weekly progress reports** in the existing `Weekly_Progress_Report.xlsx` format.
5. **Reproducibility checklist** filled out (NeurIPS-style: code, data, env, seeds, hardware).

---

## 14. Success criteria — concrete numerical targets

The project is a success if:

1. **Replication.** Pooled cross-country r² ≥ **0.55** (within 0.12 of Yeh's 0.67); mean-of-folds r² ≥ **0.60** (within 0.10 of Yeh's 0.70). Genuine replications of deep-learning papers commonly report 0.05–0.15 below the original due to unreported implementation details — this margin is honest.
2. **Per-country breakdown.** Bootstrap 95% CIs reported for all 23 countries' r².
3. **Fairness audit.** Statistical test of urban/rural gap reported with permutation p-value and Cohen's d. Whether the gap is significant or not, the analysis is the deliverable.
4. **Targeting policy claim.** Stratified targeting-accuracy table at five thresholds, with bootstrap CIs.
5. **Uncertainty-aware extension.** Per-cluster MC-dropout uncertainty estimates reported, urban-rural uncertainty gap tested with the same statistical apparatus, and the coverage-vs-rural-targeting-accuracy trade-off frontier plotted with at least 4 abstention thresholds.
6. **Temporal fairness drift.** Period-3 (held-out time) r² reported overall and stratified urban/rural, with bootstrap CI on the temporal urban-rural gap. Drift decomposition into y-shift / x-shift / relationship-shift attempted (even a partial decomposition is acceptable).
7. **Reproducibility.** Someone else can `git clone`, `conda env create -f environment.yml`, run a single command, and reproduce the headline numbers within rounding.
8. **Report and slide deck.** Polished, modeled on the diabetes paper, internally consistent with the code.

A success at level 1–4 is the "fallback" plan (matches the original solid-undergraduate scope). A success at all 8 is the full novel-extension plan that this spec commits to.

---

## 15. Why this is the right project

- It directly extends the diabetes paper's "replicate + critique + improve" arc to images.
- It introduces CNNs in a regression setting, not a yet-another-classification task.
- It produces a CV bullet that touches Stanford's Sustainability and AI Lab, two top-tier publications (*Science* + *Nature Communications*), an IJCAI fairness paper, and real-world development applications.
- It develops the multi-source data engineering muscle relevant to my aspiring data engineer career path.
- It has a defined, generous fallback (WILDS) so a bad week doesn't sink the project.
- It produces a single sharp policy claim (urban/rural targeting-accuracy gap) instead of three scattered observations.
- It is genuinely interesting work, and motivation matters for a 3-week sprint.

---

## 16. Open questions for the advisor

1. Is a written report plus slide deck the expected deliverable, or slide deck only?
2. Any preferred citation style for the report?
3. Would she like a mid-project checkpoint review at the end of Week 1?
4. Does she want a presentation rehearsal before the formal one?

---

## 17. References

### Primary papers replicated
- Jean, Burke, Xie, Davis, Lobell, Ermon. *Combining satellite imagery and machine learning to predict poverty.* Science 353(6301):790–794, 2016. DOI: 10.1126/science.aaf7894.
- Yeh, Perez, Driscoll, Azzari, Tang, Lobell, Ermon, Burke. *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa.* Nature Communications 11:2583, 2020. DOI: 10.1038/s41467-020-16185-w.

### Primary critique extended
- Aiken, Rolf, Blumenstock. *Fairness and representation in satellite-based poverty maps: Evidence of urban-rural disparities and their impacts on downstream policy.* IJCAI 2023. arXiv:2305.01783.

### Other key follow-ups
- Sarmadi et al. *Towards Explaining Satellite-Based Poverty Predictions with Convolutional Neural Networks.* IEEE 2023. arXiv:2312.00416.
- Tucker et al. *Human bias and CNNs' superior insights in satellite based poverty mapping.* Scientific Reports, 2024.
- Hall. *A review of machine learning and satellite imagery for poverty prediction.* J. International Development, 2023.
- Wang et al. *Mitigating Urban-Rural Disparities in Contrastive Representation Learning with Satellite Imagery.* arXiv:2211.08672, 2022.

### Foundational ML
- He et al. *Identity Mappings in Deep Residual Networks.* ECCV 2016 (ResNet-18 v2 with pre-activation).
- Kingma, Ba. *Adam: A method for stochastic optimization.* arXiv:1412.6980, 2014.
- Selvaraju et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.

### Benchmarks and reproducibility
- Koh et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts.* ICML 2021. arXiv:2012.07421. (PovertyMap-WILDS task.)
- Pineau et al. *Improving Reproducibility in Machine Learning Research.* JMLR 2021. (Reproducibility checklist source.)

---

*End of design.*
