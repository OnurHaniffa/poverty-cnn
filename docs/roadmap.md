# poverty-cnn — Roadmap (the agreed plan we're following)

Agreed 2026-05-25, refreshed 2026-06-09. **Ordering principle: faithful replication first → engineering/rigor improvements → novel contributions.** One GPU only (`cuda:0`; GPU 1 is the friend's).

---

## Phase 1 — Faithful replication (contribution #1)  *[nearly done]*
Reproduce Yeh 2020's 5-fold cross-country result as faithfully as possible.

- [x] Data pipeline + **stabilized baseline** — mean-of-folds **0.559** / pooled **0.569** (bootstrap 95% CI pooled [0.557, 0.582]).
- [x] **Optuna hyperparameter search** (25 trials) — best fold-A val 0.639, BUT it **did not transfer** to held-out folds (0.515). **Key finding: hyperparameter tuning does NOT beat the stabilized baseline — the model is robust, and the remaining gap to Yeh's 0.67 is DATA, not method** (we use one survey round/country ≈13.5k clusters; Yeh pooled multiple rounds ≈20k).
- [~] **Robustness seeds** — seed42=0.559, seed1=0.516 → **seed noise ≈ ±0.04** (the *dominant* uncertainty, larger than the bootstrap CI). 3rd seed (seed2) re-running. *(GPU queue)*
- [~] **Ablations at stable config:** MS-only (7ch), NL-only (1ch). *(GPU queue; old ones used the unstable config → being redone)*
- [x] **Bootstrap confidence intervals** — `scripts/11_bootstrap_ci.py`.
- [~] **Jean-2016 baseline** — DROPPED as tangential (different paper, external baseline, not needed to replicate Yeh). Light version only if time at the very end.

**Done =** replication number reported as **mean ± seed-std**, with CIs + ablations.

## Phase 1.5 — Multi-round data expansion ("pilot → full")  *[NEW, 2026-06-09]*
The single most impactful lever on replication quality, since the gap to Yeh is **data**.
Current single-round work = the validated **"pilot."** Build a parallel **"full"** dataset.

- Keep current single-round results intact (ADD, don't REPLACE).
- **116 usable GPS-era DHS surveys** available across the 23 countries (~3× data, ~40k clusters).
  Manifest: `data/raw/dhs_multiround_manifest.csv` (234 HR+GE files).
- [~] **Download** all approved DHS data — `scripts/12_download_dhs_multiround.py` (cookie-based, resumable). *Cheap, commits to nothing — gives exact cluster counts and also preps contribution #4 (temporal).*
- [ ] **Decide extraction scope** AFTER download, with real cluster counts (full ~116 vs targeted subset). Extraction = the multi-day commit (EE quota, Drive ~40-50GB, friend's PC).
- [ ] Refit PCA on pooled multi-round households → new `y` (separate clean experiment).
- [ ] Re-run pipeline + training on the "full" dataset. Expect closer to Yeh's 0.67.
- Bonus: multi-round data is exactly what contribution #4 (temporal) needs.

## Phase 2 — Improvements to the replication (rigor/engineering, *not* new science)
Make it stronger and better-reported than Yeh, without changing the science.

- **Multi-metric reporting:** R² **+ MAE + Spearman + targeting-accuracy** — not R²-only (R² conflates model error with a country's wealth spread; it misleads on its own).
- **Overfitting diagnostics:** log train-vs-val curves.
- **Nested cross-validation** for hyperparameter selection (professor-preferred rigor; ~5× compute — deliberate).
- **Country/group + urban/rural rebalancing** (loss reweighting / balanced sampling) — see `docs/future-work.md`. NB: also feeds the fairness story — audit the *unweighted* model first (the disparity is the finding), then reweight as a *mitigation*.
- **Pretrained-adapted ResNet** ablation (does ImageNet init help vs from-scratch?).

## Phase 3 — Novel contributions (the new science)
1. **Fairness audit (contribution #2):** per-country + urban/rural across all 23, **multi-metric**, with the across-seed noise floor. Extends Aiken 2023 (10 → 23 countries).
2. **Uncertainty (contribution #3):** MC-dropout (dropout already baked into the model) + uncertainty-aware **aid-targeting simulation**.
3. **OOD generalization (banked countries):** South Africa, Namibia, Gabon, Eswatini, Madagascar, Niger, Liberia.
   - Protocol: **project onto the FROZEN 23-country PCA axis (never refit)** → extract tiles **only after the final tuned model exists** → run the **frozen trained model** on them as a **TEST set only (never train/val).**
   - Expected to degrade (asset-index ceiling in wealthier countries) — **that limitation is the finding.**
4. **Temporal drift (contribution #4):** train earlier composite windows, test later; does the urban/rural gap widen over time?
   - **Now UNLOCKED by Phase 1.5** — the multi-round download provides the same-country multi-year data this needs.
   - Scope tightly to countries with good multi-round coverage (e.g. Senegal's continuous DHS 2012-2019).
   - ⚠️ Do *not* fake it with single-round data (would confound time with country).

## Standing decisions
- **GPU:** `cuda:0` only — GPU 1 is the friend's, never touch it.
- **Multi-round (Phase 1.5):** download all approved DHS now (cheap); decide extraction scope after, with real counts; keep single-round results as the validated pilot (ADD, don't REPLACE).
- **OOD countries:** test-only; frozen model + frozen PCA; extract after the final model.
- **Tuning:** done — found the model is robust / gap is data-bound; don't re-litigate.
- **Commits:** held until the user says so.
