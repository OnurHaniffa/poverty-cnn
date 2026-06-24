# Methodology Defense — answers to the questions the committee will ask

Prep brief for the lab presentation. For each likely question: the short answer, our
evidence, and the honest fallback. We are *more* rigorous than the original paper —
this is organizing our armor, not patching gaps.

## Q1 — Overfitting: prevent & detect
**Short answer:** the whole evaluation is built against it, and we measure the train–test gap directly.
- **Prevent:** (1) cross-country CV — a model that memorises a country scores ~0 on unseen
  countries; we get 0.60, so it generalises; (2) early stopping on val r² (best-val checkpoint);
  (3) dropout 0.2; (4) label-preserving augmentation (flips/90° rotations); (5) weight decay.
- **Detect (full model, our numbers):** train r² **0.70–0.80** (NOT ~1.0 → no memorisation).
  Train–test gap tiny for folds A/C/D (≤0.08), larger for B/E (0.22, 0.33) — the latter is
  *harder test-country groups*, documented by the per-country audit, not gross overfitting.
  Also: 5-seed stability + bootstrap CIs.
- **If pushed:** the held-out cross-country r² IS the overfitting test — it can't be gamed by memorising.

## Q2 — Imbalanced data
**Short answer:** we identified three imbalances, tested the standard fix, and rigorously showed its limits.
- **Imbalances:** skewed wealth tails (an imbalanced-regression problem, Yang 2021 ICML);
  country sample-size (Kenya 1585 vs Mali 328); urban/rural (~8.6k vs ~5k).
- **What we did:** tested LDS-style reweighting (symmetric + asymmetric, α-sweep, 5-fold) →
  it does NOT repair the poorest-bias (it's a signal limitation, not a loss artifact). Country
  imbalance handled by cross-country CV (equal weight) + a per-country audit (no hiding small
  countries behind a pooled mean).
- **If pushed:** "We treated imbalance as a hypothesis and falsified the easy fix — which is
  WHY we conclude the failure is fundamental."

## Q3 — Metrics
- r² + MAE + Spearman (r² alone conflates error with variance — we argue this explicitly).
- Added: RMSE **0.528**; poorest-20% targeting precision/recall/F1 **48.8%** (2.4× random).
- Per-subgroup (urban/rural, per-country) + bootstrap CIs + 5-seed variance.
- **Uncertainty evaluated with the field-standard metrics** (not correlation — the deep-regression
  UQ literature discourages corr(unc,error) in favour of these):
  - **AUSE / AURG** (sparsification): AURG = MC **−0.003** (≈random), ensemble **+0.046**, hetero
    **+0.083** — all far from oracle → uncertainty barely ranks errors.
  - **Calibration** (reliability diagram, ENCE, PICP@90, variance-scale s): epistemic-only methods
    are overconfident (ensemble PICP90 0.33 / s 5.8; MC 0.20 / s 8.6) — they capture *epistemic*
    variance but miss the dominant *aleatoric* noise. The **heteroscedastic** head (learns the noise)
    is genuinely well-calibrated: ENCE **0.025**, PICP90 **0.866**, s **1.17**, NLL **0.88**.
  - **Equity (group-conditional coverage @90%)**: epistemic give the poor *worse* coverage
    (.12 vs .39); the calibrated hetero gives *equal* coverage (.84 vs .83) → it does NOT widen its
    bars for the poor. **Even a well-calibrated model can't flag the poorest-bias — calibration ≠ equity.**
  - Figures: `results/figures/teaching/uq_sparsification.png`, `uq_calibration.png`.

## Q4 — Validation honesty / leakage
- Country-disjoint folds (Yeh Supp Table S2). Leakage-free normalisation (train-country stats
  only). Pilot→full design; temporal & OOD holdouts for external validity.
- **Whole-country (cluster) bootstrap** = the honest CI: because the signal is between≫within,
  the i.i.d. bootstrap is too narrow. Resampling the 23 countries gives r² 0.606
  **[0.538, 0.658]** (≈9× wider than the naive [0.599, 0.612]) — that's the headline CI we report.
- **Multiple-comparison control:** all **23/23** per-country Spearman tests survive
  Benjamini-Hochberg FDR at q<0.05 (ρ 0.56–0.84). No p-hacking across the family.
- **Spatial-autocorrelation defence — MEASURED on our own data (`scripts/47`):** a naïve random
  80/20 cluster split (countries mixed across train/test) scores r² **0.729** vs our country-blocked
  **0.606** — a **+20% inflation**, right in line with the literature's "up to ~28%" (Kattenborn 2022).
  Whole-country blocking is the *strongest* spatial blocking and *cannot* leak nearby clusters, so
  our 0.606 is the honest number; the 0.73 we *refuse to claim* is what naive CV would have inflated to.
- **Residual SAC check (`scripts/48`):** held-out residuals show moderate within-country spatial
  autocorrelation (Moran's I ≈ 0.28, mean over 23 countries). Honest reading: this is *unmodelled
  local structure* (a per-tile CNN misses regional effects — future work), **not leakage** (leakage
  needs train clusters near test clusters, impossible across country-disjoint folds). It also
  *justifies* the country-level bootstrap above: within-country clusters are not independent.
- **Honest limitation:** no full nested CV (Optuna used one val split) — BUT we have direct
  evidence it matters: tuning overfit that split (0.64 val → 0.52 test). So single-split tuning
  is shown to overfit; nested CV is the fix (~5× compute), a documented limitation.

## Q5 — Preprocessing & EDA
- Full EDA (distributions, asset prevalence, wealth-index validation, per-country maps).
- Documented preprocessing: PCA wealth index; tile crop/NaN-fill/normalisation; trainability
  filter; special-code handling; null-island (missing-GPS) exclusion.
- Documented divergences (single-round pilot; asset collapsing) — stated, not hidden.

## Q6 — Generalisation / external validity
- Internal: cross-country CV (0.60 on unseen countries).
- Temporal: train-early/test-late holdout (running).
- OOD: frozen model on wealthier, never-seen countries (planned) — the "where does it break" test.

## Q7 — Limitations we VOLUNTEER (disarms the hard questions)
- From-scratch — but we tested it: an ImageNet-pretrained ResNet-18 (fold A) scored r² 0.663 vs
  from-scratch 0.690 (**−0.027, pretraining doesn't help**), so the gap to Yeh is DATA, not a
  from-scratch penalty (ImageNet→satellite transfer is weak; SatMAE-style pretraining = future work).
- No formal multiple-comparison correction across the 23 per-country tests (partly handled by the
  cross-seed noise floor + CIs).
- Residual ~0.10 r² gap to Yeh: shown to be DATA (pretraining doesn't help, −0.027), plus pooled-PCA
  comparability — root cause measured (night-lights flat-dark for the poorest half, `scripts/49`).

## The headline numbers to have memorised
- Replication: pooled r² 0.569 (pilot) → **0.606 (full)** [honest CI 0.538–0.658]; mean-of-folds 0.517 → **0.597**.
- **Benchmark reframe (the strong one):** in **Pearson r** — the metric WILDS PovertyMap reports —
  we hit **0.78 = the published benchmark (Koh 2021)**, and worst-group r **0.55 > their 0.45**, on the
  **same** leave-country-out (OOD) protocol on the same DHS data. The "0.10 r² gap to Yeh" is the
  from-scratch-vs-pretraining penalty in r²-framing, not an accuracy deficit in the field's metric.
- Fairness persists at 3× data (poorest-bias +0.62 → +0.59). Uncertainty can't flag it (even
  well-calibrated: equal poor/rich coverage). Reweighting can't fix it.
- The thesis = **3 independent legs** (confidently-wrong → undetectable → resists 2 mitigation families)
  **+ 2 regime-replications** (holds at 3× data, and recurs OOD on the rich tail). Root cause MEASURED
  (night-lights flat-dark below the poverty line). Don't say "5 independent ways" — in-dist & OOD are
  the same regression-to-mean in two regimes.
