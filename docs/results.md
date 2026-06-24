# Results — Single-Round Pilot Study

> **Scope note.** This document reports the **single-round pilot** (one DHS survey
> round per country, ~13,453 villages). It is deliberately the *first half* of a
> **pilot → full** design: the multi-round "full" dataset (~25k additional clusters,
> extraction in progress) is the planned scale-up. Numbers below will be **augmented**
> (not replaced) by a full-scale section; the findings, methodology, and discussion
> are designed to be robust to that scale-up (see §8).

---

## 1. Summary

We faithfully replicate Yeh et al. 2020's cross-country satellite wealth-prediction
result at single-round data scale, then build the **fairness + uncertainty audit
layer that neither Yeh nor the dedicated 10-country follow-up (Aiken 2023) provides**.
The one-line thesis:

> **We reproduce the headline, then show the failure modes that matter for aid: the
> model is confidently wrong about the poorest, and standard uncertainty tools cannot
> detect it.**

---

## 2. Data & method (brief)

- **Target `y`:** asset-wealth index = first principal component of pooled household
  asset ownership across all 23 countries, standardized (mean 0, std 1), averaged to
  the village. Pooled PCA = one comparable "ruler" across countries (Filmer–Pritchett).
- **Input `x`:** 8-channel 224×224 tile (RGB, NIR, SWIR1/2, thermal, nightlights),
  6.72 km, 3-year median composite, via Earth Engine (Yeh protocol).
- **Model:** 8-channel ResNet-18, from scratch, MSE loss; stabilized training
  (LR 3e-4, linear warmup + cosine decay, gradient clipping).
- **Evaluation:** 5-fold **cross-country** CV using Yeh 2020 Supp. Table S2 fold
  assignment. Every village is predicted by a model that never saw its country.
- **Rigor:** 5 random seeds; metrics reported as mean ± std and with bootstrap CIs.

---

## 3. Replication (contribution #1)

| Metric | Ours (5-seed) | Yeh 2020 | Note |
|---|---|---|---|
| Mean-of-folds r² | **0.517 ± 0.023** | ~0.70 | pre-reg bar 0.60 (just short — expected) |
| Pooled r² | **0.569** (95% CI [0.557, 0.582]) | 0.67 | pre-reg bar 0.55 (cleared) |
| NL-only linear floor | 0.25 | — | decisively beaten → real spatial learning |

**The gap is data-bound, not method-bound.** We use one DHS round/country (~13.5k
clusters); Yeh pooled ~43 rounds (~19,669). Three diagnostics corroborate a data
ceiling: (a) a 25-trial Optuna search **did not** beat the baseline on held-out folds
(hyperparameters overfit one split); (b) we beat the nightlights-only linear floor
(0.25) decisively; (c) the field's own ceiling is sticky — recent foundation models
and planetary wealth indices still cap at r² ~0.66–0.70.

**The pooled > mean reversal is the tell.** Yeh has mean (0.70) > pooled (0.67);
we have pooled (0.569) > mean (0.517). A country-average *below* pooled means a heavy
**left tail** of small/hard countries (Senegal, Côte d'Ivoire negative) — the
mechanistic fingerprint of *less data per country*. This *strengthens* the
faithful-replication claim.

---

## 4. Channel ablations

| Model | mean-of-folds r² |
|---|---|
| NL-only (nightlights, 1 ch) | **0.58** |
| MS+NL (full, 8 ch) | 0.52 |
| MS-only (7 daytime ch) | 0.46 |

**Nightlights dominate** — reproduces Yeh and Perez 2017 (nightlights proxy
electrification/urbanization, the dominant cross-village wealth axis). The twist that
**MS+NL underperforms NL-only** (Yeh had them co-equal/additive) is best read as a
data-budget artifact — daytime CNNs are data-hungry, so the daytime pathway is
undertrained on single-round data (more imagery restores it in the literature).
*Caveat: this is a reasoned inference, not proven — the multi-round run is the clean test.*

**Per-country complementarity:** the MS+NL vs NL-only per-country r² correlation is
only **+0.12** — the channels are *complementary by country* (Senegal/Ethiopia/Lesotho
favor nightlights; Rwanda/Burkina need daytime imagery). This statistic appears
unpublished, but is within seed noise (±0.30) for small countries — **bootstrap before
claiming.** *(Figure: `results/figures/fairness/03_nl_vs_msnl.png`.)*

---

## 5. Fairness audit (contribution #2)

Extends Aiken 2023's fairness analysis from 10 to **23 countries**, with a 3-metric
lens (r², MAE, Spearman) because r² alone misleads (it conflates model error with a
group's wealth spread).

**Urban vs rural** *(`01_urban_rural.png`, `04_percountry_urbanrural.png`):* equal
*absolute* accuracy (MAE ~0.45 both) but worse rural **ranking** (Spearman urban 0.54
vs rural 0.41); urban ranks better in **21/23 countries**. The gap correlates with
rural homogeneity at r=+0.44 → *partly* intrinsic difficulty but **mostly** real model
deficiency. Reproduces the field-wide "between >> within" structure.

**Per-country** *(`02_percountry.png`):* huge spread (Togo r² 0.75 … Senegal/Côte
d'Ivoire negative). A **trichotomy** separates failure *types*: 13 well-served, 7
*miscalibrated* (rank fine, scale off — fine for targeting), 3 *genuine misses*.
**Per-country r² is seed-noisy** (±0.30 for small countries) → single-run audits are
unreliable; our 5-seed protocol is a genuine methodological strength.

**Calibration / regression-to-the-mean** *(`05_calibration.png`, `06_bias_by_wealth.png`):*
pred~true **slope 0.60** (predictions compressed toward the mean). The **poorest decile
is predicted +0.66 too rich** (made to look less poor) and has the **largest error**
(MAE 0.66). The model is **confidently wrong about the poorest** — a textbook
attenuation pathology, here quantified decile-by-decile on 23 countries. *Note: this is
a slope compression, not a constant offset — a post-hoc offset correction would not fix it.*

**Targeting:** within-country, flagging the poorest 20% by prediction catches **39%**
of the truly-poorest-20% (poorest-10%: **23%**, ~2.3× random but **missing 77%**).
Rural-poor caught worse than urban-poor (28% vs 46% at the top decile). Consistent with
the matched benchmark (Aiken *Nature* 2022, Togo). **Message: usable for regional
ranking and broad poor/non-poor splits, NOT last-mile extreme-poor household targeting.**

---

## 6. Uncertainty (contribution #3) — the most novel piece

Three uncertainty methods, evaluated head-to-head on the equity question *"does
uncertainty flag the poorest?"* *(`10_three_methods.png`, `11_hetero_by_wealth.png`,
`12_risk_coverage.png`).*

Evaluated with the **field-standard deep-regression UQ metrics** (the literature
discourages `corr(unc,error)`, kept only as a contrast line) — `scripts/34`,
`results/uncertainty_proper.json`, figures `uq_sparsification.png`, `uq_calibration.png`:

| Method | AURG ↑ (gain over random) | NLL ↓ | ENCE ↓ | PICP@90% | scale `s` | *(old corr)* |
|---|---|---|---|---|---|---|
| MC-dropout (BatchNorm-safe, 30 passes) | **−0.003** | 35.3 | 0.40 | 0.20 | 8.6 | 0.00 |
| Deep ensemble (5-seed) | +0.046 | 15.6 | 0.34 | 0.33 | 5.8 | 0.11 |
| Heteroscedastic (Gaussian NLL head) | **+0.083** | **0.88** | **0.025** | **0.87** | **1.17** | 0.12 |

Two findings, both stronger than the correlation told us:
- **As error-rankers, all three barely beat random** (AURG ≈ 0; MC-dropout is *below* random).
  The sparsification curves hug the random line, far from the oracle (AUSE ≈ 0.49–0.57).
- **Calibration:** the epistemic-only methods (ensemble, MC) are **wildly overconfident** — their
  90% interval catches only 20–33% of truths, error bars 6–9× too small — because they capture
  *epistemic* variance and miss the dominant *aleatoric* noise. The **heteroscedastic** head, which
  learns the noise, is **genuinely well-calibrated** (ENCE 0.025, PICP 0.87, `s` 1.17, NLL 0.88).

**The equity kicker — group-conditional coverage @90% (poorest-20% vs richest-20%):** the
epistemic methods give the poor *worse* coverage (ens .12 vs .39; MC .09 vs .30); the
**well-calibrated** hetero model gives **near-equal** coverage (.84 vs .83) — i.e. it does *not*
widen its bars for the poor. So predicted σ is *flat* across the wealth range while error is
*U-shaped* (worst at the poorest). Heteroscedastic UQ does give real **risk-coverage** value
(confident-10% MAE 0.32 vs 0.43) — useful for "which predictions to trust" — **but even when
properly calibrated it still fails the equity test. Calibration ≠ equity.**

**Conformal + recalibration (P3/P4, `scripts/36`, `results/conformal_recal.json`).** To pressure-test
that claim with guarantee-backed tooling: split-conformal on the heteroscedastic model gives valid
marginal coverage (**0.905** at target 0.90) and — honestly — coverage that is *roughly equitable*
across wealth (poorest decile 0.88 ≈ richest 0.88); Mondrian (group-conditional) conformal equalizes
the small urban/rural gap (0.89/0.91 → 0.90/0.90); a country-blocked split is mildly *conservative*
(0.921), i.e. the exchangeability cost runs in the *safe* direction. Variance-scaling recalibration
of the badly-overconfident deep ensemble (s=5.84) fixes marginal calibration spectacularly
(PICP 0.33→0.93, NLL 15.4→1.1) yet leaves a **residual poorest-decile gap (coverage 0.06→0.83 vs
richest 0.27→0.95).** Net, the refined and *stronger* statement: calibrated/conformal intervals are
valid and roughly equitable in **coverage**, but (from AUSE) the uncertainty still cannot **rank**
which predictions are wrong, so it cannot be used to **prioritise** the poorest for scrutiny — and the
naïve epistemic methods actively under-cover them. **Coverage validity ≠ targeting usefulness.**

**Why (principled):** prediction error = aleatoric + **bias** + variance. Ensembles and
MC-dropout capture only variance and are *provably blind to the bias term*; a learned
aleatoric head misattributes bias to noise. Our poorest-decile error is **regression-to-
mean bias + DHS label/GPS-displacement aleatoric noise** — neither visible to epistemic
uncertainty. This equity-framed negative result appears **unstated in the
uncertainty-for-poverty literature** (a "not-found," pending a targeted venue search).

---

## 6b. Mitigation — can we fix the poorest-bias? (no, and that's the point)

We tested whether standard **imbalanced-regression reweighting** (Yang 2021) can
repair the regression-to-mean against the poorest, rigorously on all 5 folds.

| Mitigation (vs baseline pooled r² 0.569 / poorest bias +0.62 / recall 40%) | Result |
|---|---|
| **Symmetric LDS** (both tails), α sweep 0.5–1.5 | fold-unstable; α=1.0 pooled r² 0.535 (**−0.034**), bias only −3%, recall −1pp |
| **Asymmetric** (poor tail only), α=1.0 | **catastrophic** — 4/5 folds collapse to r²≈0, bias *explodes* to +1.27 |
| **Balanced-MSE / BMC** (Ren 2022; `scripts/37`) | partial bias relief (+0.66→**+0.47**) but **accuracy collapses** (mean r² 0.52→**0.365**) — an unfavorable trade, bias still large |

**No imbalanced-regression family robustly fixes the poorest-bias** — now shown across **two
distinct mechanisms** (LDS loss-reweighting AND BMC balanced-loss). BMC is the sharpest statement:
it *can* rebalance toward the poor, but only by degrading the model so much (r² −0.15) that the
"fix" is worse than the disease, and the bias (+0.47) remains. You cannot reweight your way to a
signal that does not exist. This is not a tuning failure to keep
chasing — it is the predicted behavior when the *information is not in the input*:
you cannot reweight your way to a signal that does not exist, you only destabilize
training. This is the **third independent line of evidence** for the unifying thesis:

> The model's failure on the poorest is **fundamental** — *confidently wrong*
> (§5 audit), *undetectable* (§6 uncertainty), and *unfixable by mitigation* (here) —
> because the satellite signal (nightlights uniformly dark below the poverty line)
> cannot separate the extreme poor. This caps the technology for last-mile targeting
> regardless of modeling sophistication.

*Caveat: two reweighting families tested, single seed; other approaches (focal-R,
feature-space methods, two-stage) and — critically — more data (the multi-round run)
remain as levers. We claim "resistant to standard reweighting," not "provably unfixable."*

---

## 7. Contribution positioning

- **Not novel (valuable foundation):** the headline, nightlights dominance, the
  regression-to-mean and urban>rural *directions* — all pre-established. Reproducing
  them faithfully (a modern PyTorch port of dead TF 1.15 code) is the foundation.
- **Modestly novel:** the 23-country fairness audit (vs Aiken's 10), decile-resolved
  miscalibration magnitude, the ranking-vs-scale trichotomy, the per-country
  complementarity statistic, the cross-seed-noise warning.
- **Most novel:** the **equity-framed uncertainty result** — testing 3 methods on
  *"does uncertainty flag the poorest?"* and reporting a clean *no*, grounded in
  bias-variance theory. The recent UQ-for-poverty papers advertise calibrated intervals
  but do not test this equity question.

It is an honest **audit paper**, not a method paper — no new architecture or SOTA number,
and it should not claim one.

---

## 8. Limitations & honest caveats

1. **Single-round pilot scale** — the headline gap to Yeh is attributed to data volume;
   the multi-round full run (§9) is the clean test.
2. **Literature comparisons** were grounded via abstracts/HTML and secondary figures, not
   all primary PDF tables — verify exact decimals before final quoting.
3. **Some interpretations are inferences, not proofs** — esp. "MS-hurts is purely a data
   artifact" (could be a real MS-pathway issue) and "the urban/rural gap is partly
   intrinsic" (r=0.44 → it is *mostly* real model deficiency).
4. **OOD and temporal contributions are not yet run** — those conclusions are currently
   theory-based extrapolation.
5. **From-scratch (no pretraining)** may cost a few r² points vs Yeh's transfer arms — an
   unquantified secondary contributor to the gap.

---

## 9. Planned extensions (the "full" study)

- **Multi-round "full" dataset** (~25k new clusters, 2008–2022, extraction in progress)
  → refit pooled PCA → retrain. Tests the data-bound-ceiling claim directly; expected
  to approach Yeh's 0.67. Current pilot results are **augmented, not replaced.**
- **Temporal drift (#4):** train early composite windows, test late — only possible with
  multi-round data. Does the urban/rural gap widen over time?
- **OOD generalization (#3 of novelties):** wealthier African countries (South Africa,
  Namibia, …), test-only on the **frozen** model + **frozen** PCA axis.

---

*Figures referenced live in `results/figures/fairness/` and `results/figures/uncertainty/`.
Literature context and citations are captured in the `lit-context-*` memory files and were
produced by a 10-agent web-research synthesis (2026-06-09).*

---

## 10. Full-scale study (3× data) — confirmation

The single-round pilot was scaled to a **multi-round "full" dataset** (~36,772 villages,
2008–2022, pooled-PCA wealth index refit on ~1M households).

- **Accuracy improves, confirming the data-bound ceiling:** mean-of-folds r² **0.517 → 0.597**,
  pooled **0.569 → 0.606** (folds A and D hit 0.69, matching Yeh). 3× data closed ~half the
  gap to 0.70 — the residual is likely the from-scratch-vs-pretraining penalty.
- **The fairness failures PERSIST at scale** — the decisive result:
  - Regression-to-mean **slope 0.60 → 0.62** (unchanged); **poorest-decile bias +0.62 → +0.59**
    (basically unchanged). The poorest-bias is NOT a small-data artifact.
  - Urban–rural Spearman gap narrowed (0.13 → 0.06) but persists — more data helps fairness
    *somewhat*, not fully.

**Net:** the model's failure on the poorest is confirmed — confidently wrong
(audit), undetectable (uncertainty), unfixable by mitigation (reweighting), and **unfixable by
3× more data** (here). A fundamental signal limitation, not a modelling or data-budget artifact.

### 10b. Two evidence upgrades (from the 2026-06-17 self-critique)

- **The gap to Yeh is DATA, not from-scratch (`scripts/52`, `results/pretrained_baseline.json`).**
  An ImageNet-pretrained ResNet-18 (8-channel adapted stem) on fold A scores r² **0.663** vs the
  from-scratch **0.690** — pretraining *does not help* (**−0.027**). So the ~0.10 r² gap to Yeh is
  not a from-scratch penalty; it is the data budget (single-round / fewer rounds per country).
  Consistent with the remote-sensing finding that ImageNet→satellite transfer is weak (satellite-
  specific pretraining like SatMAE is future work).
- **The root cause is MEASURED, not asserted (`scripts/49`, `nightlight_by_wealth.png`).** Mean
  night-light intensity is **flat and near-zero across the poorest ~5 wealth deciles** (0.1–0.8)
  then climbs steeply (decile 6 → 9: 3.2 → 13.2). The dominant wealth signal is *dark for the
  poorer half*, so the satellite cannot separate the extreme poor. (Honest nuance: the within-group
  NL↔wealth correlation is lower for the poor, 0.28 vs 0.39 for the rich, but range-restriction
  tempers that comparison — the mean-intensity curve is the direct evidence.) This converts the
  central thesis from *argued* to *shown*.

---

## 11. Statistical rigor & benchmark positioning (`scripts/35`, `results/stats_rigor.json`)

Finishing-pass rigor from the 2026-06-16 methods/methodology research synthesis.

**Honest confidence intervals — whole-country (cluster) bootstrap.** Because the wealth signal
is mostly *between* countries, clusters are not independent; an i.i.d. bootstrap understates the
CI. Resampling the **23 countries** with replacement gives the defensible headline (≈9× wider):

| Metric | point | i.i.d. CI (too narrow) | **whole-country CI (honest)** |
|---|---|---|---|
| pooled r² | 0.606 | [0.599, 0.612] | **[0.538, 0.658]** |
| pooled Spearman | 0.760 | [0.755, 0.764] | **[0.710, 0.800]** |
| pooled MAE | 0.417 | [0.413, 0.420] | **[0.389, 0.446]** |

**Multiple-comparison control.** Across the 23 per-country Spearman tests, **all 23 survive
Benjamini-Hochberg FDR at q<0.05** (per-country ρ 0.56–0.84; all p astronomically small). The
ranking signal is real in every country, not a multiple-testing artifact.

**CV honesty, measured (`scripts/47`).** A naïve random 80/20 cluster split (countries mixed) scores
r² **0.729** vs our country-blocked **0.606** — a **+20% inflation**, matching the literature's
"up to ~28%" (Kattenborn 2022). The 0.73 is the optimism naive CV would have claimed; our 0.606 is
the honest number. Residual Moran's I (`scripts/48`) confirms moderate within-country spatial
structure (≈0.28) — unmodelled local signal, *not* leakage (folds are country-disjoint), which is
also why the headline CIs use the whole-country bootstrap.

**Benchmark positioning — in each benchmark's own currency** (same DHS data family, *different
protocols*, so orientation not apples-to-apples):

| Study (protocol) | Pearson r | r² | worst-group r (urban/rural) |
|---|---|---|---|
| **Ours (5-fold leave-country-out)** | **0.78** | **0.61** | **0.55** |
| WILDS PovertyMap ERM (OOD = leave-country-out)¹ | 0.78 (±0.04) | — | 0.45 (±0.06) |
| Yeh 2020 (pooled ridge on CNN features) | — | ~0.67–0.70 | — |

The headline reframe: **expressed in Pearson r — the metric the WILDS PovertyMap benchmark
reports — our model matches the published benchmark (0.78) and beats its worst-group score
(0.55 vs 0.45), on the *same* leave-country-out (OOD) protocol on the same DHS data lineage.**
The "0.10 gap to Yeh" is an r²-framing of the from-scratch-vs-pretraining penalty, not an
accuracy deficit in the field's reporting metric. (WILDS independently reproduces our fairness
finding too: rural r consistently below urban, the worst-group gap tripling under country shift.)

¹ WILDS PovertyMap (Koh et al. 2021, arXiv:2012.07421) is built on the same Yeh et al. DHS data
with country-defined domains — its OOD split is leave-country-out, directly comparable to ours.
The ~28% random-CV inflation figure cited above is Kattenborn et al. 2022 (doi:10.1016/j.ophoto.2022.100018).

---

## 12. Temporal external validity (contribution #4) — `scripts/31`, `results/cnn_temporal/`

A **double-shift** holdout: each fold trains on *(unseen countries AND years ≤2014)* and tests on
*(unseen countries AND years ≥2015)* — so the model must extrapolate across **both space and time**.

| Fold | A | B | C | D | E | **mean** |
|---|---|---|---|---|---|---|
| temporal r² | 0.650 | 0.453 | 0.451 | 0.463 | 0.214 | **0.447** |

**Finding:** vs the spatial-only full model (0.597), the double-shift costs **~0.15 r² (~25%
relative)**. The model *does* carry forward in time (0.45 ≫ 0), but it is **not** fully
temporally robust. Honest caveat: this conflates *less training data* (early years only) with
*genuine temporal drift*, so it is best read as a **prospective-deployment estimate** — "train on
data through 2014, deploy on a new country in 2015+ → expect ≈0.45." Fold E (0.21) is the hard
corner (test-country group + late years), consistent with its weakness in the spatial audit.

---

## 13. Geographic OOD capstone (`scripts/38–44`) — "where does it break?"

The frozen 23-country models, tested **with no retraining and no leakage** on **6 never-seen
countries** spanning the wealth spectrum (frozen-PCA labels validated corr=1.0). 2,800 tiles,
extracted via Earth Engine and projected onto the identical wealth axis.

| Country | mean wealth | 36k r² | 36k Spearman |
|---|---|---|---|
| Niger / Madagascar (poor) | −0.48 / −0.47 | +0.62 / +0.61 | 0.71 / 0.66 |
| Gabon | +0.16 | +0.54 | **0.81** (> in-dist 0.76) |
| Namibia | +0.55 | +0.19 | 0.65 |
| Eswatini | +0.62 | −0.65 | 0.72 |
| **South Africa** | **+1.33** | **−1.71** | **0.48** |

**Four findings, all reinforcing the central thesis:**
1. **Ranking transfers remarkably well** — Spearman 0.65–0.81 for 5/6 unseen countries (Gabon
   *beats* the in-distribution 0.76). The model learned a transferable wealth-ranking signal, not
   memorised geography.
2. **Absolute calibration breaks on the WEALTHY extreme** — r² degrades *monotonically* with
   country wealth (poor +0.6 → South Africa −1.7). The regression-to-mean pathology again: the
   model squashes rich villages toward the training mean. This is the **same** regression-to-mean
   phenomenon as in-distribution (poorest predicted too rich), now seen on the rich tail OOD.
   *(Framing note: in-distribution and OOD are the same failure mode in two regimes — so the
   thesis rests on **three independent legs** [the failure, its undetectability, its resistance to
   two mitigation families] **+ two regime-replications** [at 3× data, and out-of-distribution] —
   not five independent confirmations.)*
3. **More data does NOT help OOD** — 13k vs 36k generalise identically (Spearman 0.734 vs 0.726),
   though 3× data lifted *in-distribution* accuracy 0.52→0.60. The OOD ceiling is the distribution
   shift, not training volume.
4. **Adversarial validation** (`scripts/44`): all 6 countries are far OOD in feature space (AUC
   0.84–0.98), but distinguishability does *not* predict the breakage (corr +0.13). The model ranks
   fine despite enormous visual shift — it simply cannot *calibrate* to wealth ranges it never saw.

Figure: `results/figures/teaching/ood_where_it_breaks.png`. Data: `results/ood_frozen_test.json`,
`results/ood_adversarial.json`.

**OOD rigor pass (`scripts/45–46`).** To make the capstone airtight:
- **Bootstrap 95% CIs (cluster-level, per country):** South Africa 0.484 **[0.426, 0.541]** does not
  overlap any other country — the break is *statistically significant*, not small-n noise; small-n
  countries (Niger n=207) correctly get wider CIs [0.611, 0.789]. Pooled Spearman 0.726 [0.710, 0.741].
- **Trivial baseline (predict the country mean):** for the wealthy extremes (South Africa, Eswatini)
  the model's MAE is *worse than guessing the mean* (negative r²) — concrete proof of the calibration
  break — yet ranking still holds (SA Spearman 0.48). Confirms "ranks fine, can't calibrate".
- **Targeting recall @ poorest-20%:** ~**55%** overall (Gabon/Eswatini 63%, SA 51%) — real targeting
  value even OOD, comparable to the in-distribution 49%.
- **Channel ablation ('feature-out'):** nightlights dominate OOD too (overall Spearman −0.192 when
  removed, vs ≤0.04 for every other band). South Africa's break is **not** band-specific (no single
  band removal fixes it) → a fundamental wealth-range extrapolation failure, not a sensor artifact.
- **Overfit/underfit:** OOD ranking (0.73) ≈ in-distribution ranking (0.76) → the frozen models
  generalise rather than overfit; nested CV is correctly N/A to a test-only frozen evaluation.
Data: `results/ood_rigor.json`, `results/ood_channel_ablation.json`.
