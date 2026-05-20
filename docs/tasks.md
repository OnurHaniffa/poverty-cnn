# Project tasks

Live tracking of where the project stands. Updated as work progresses.

See [`design.md`](design.md) for the full design and timeline, and
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the day-by-day narrative.

## Status

| Phase | Status | Notes |
|---|---|---|
| Day 1 — Bootstrap | ✅ Done | Repo, env, approvals, GitHub all complete |
| Week 1 — Data and infrastructure (May 8–15) | 🟡 In progress | Hard checkpoint May 15 (today) |
| Week 2 — Replication and main result (May 16–22) | ⏳ Pending | Hard checkpoint May 22 |
| Week 3 — Fairness audit + extensions + write-up (May 23–30) | ⏳ Pending | |

## Day 1 — bootstrap (✅ done)

- [x] Create `poverty-cnn/` repo skeleton
- [x] Write `environment.yml`, `README.md`, `.gitignore`, `pyproject.toml`, `LICENSE`
- [x] Copy spec to `docs/design.md`
- [x] Initialize git, first commit
- [x] Create GitHub repo (public), push — `github.com/OnurHaniffa/poverty-cnn`
- [x] Register for DHS data — **approved, all 23 countries (Survey + GPS)**
- [x] Register for Google Earth Engine — **approved, project `storied-chimera-491721-i4`**
- [x] Create local conda environment, run smoke test
- [ ] Configure VS Code Remote-SSH to school server — **still pending, machine was powered off**

## Week 1 — data and infrastructure (May 8–15)

| Day | Date | Task | Status |
|---|---|---|---|
| 1 | May 9 (Sat) | Repo skeleton, registrations, env setup | ✅ |
| 2 | May 10 (Sun) | Reading list (`africa_poverty`, `predicting-poverty-replication`, Aiken 2023). Launch background EE download. | 🟡 reading done; **bulk EE extraction not yet launched** |
| 3 | May 11 (Mon) | Pull DHS for all 23 countries; pooled PCA wealth index; verify against WILDS | 🟡 wealth-index code done; only Kenya downloaded so far |
| 4 | May 12 (Tue) | DHS + EE approvals come through; env fixes (numpy<2, pip retries) | ✅ |
| 5 | May 13 (Wed) | Data pipeline modules written + Kenya end-to-end tested (1,594 clusters, 0.99σ urban/rural gap, 2 tiles downloaded) | ✅ |
| 6 | May 14 (Thu) | CLAUDE.md + PROJECT_CONTEXT.md added for session continuity | ✅ |
| 7 | May 15 (Fri) | **Hard checkpoint day.** Outstanding: SSH server access, bulk EE extraction, `dataset.py`, `splits.py`, WILDS parity check | 🟡 in progress |

## Week 2 — replication and main result (May 16–22)

See `docs/design.md` §11 for the full Week 2 day-by-day. Headline tasks:
hyperparameter search → 5-fold cross-country training → MC-dropout inference → ablations → Jean 2016 baseline → temporal training kickoff.

## Week 3 — fairness audit + extensions + write-up (May 23–30)

See `docs/design.md` §11 for the full Week 3 day-by-day. Headline tasks:
fairness audit → uncertainty-aware extension → temporal drift analysis → visualizations → final report + slide deck.

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | Picked Yeh 2020 (Nature Comms) replication over PlantVillage / HAM10000 / PCam | Top venue, real-world impact, fits aspiring-data-engineer career path |
| 2026-05-09 | 5-fold cross-country protocol matching Yeh's Supp Table S2 | Apples-to-apples comparison; standard CV; not 7/8/exhaustive |
| 2026-05-09 | Added uncertainty-aware fairness extension as novel contribution | Real research novelty; cheap (no extra training) |
| 2026-05-09 | Added temporal fairness drift extension as novel contribution | Real research novelty; ~5–10 GPU-h additional |
| 2026-05-09 | Marked bias-amplification simulation as Week 3 stretch | Highest novelty but requires simulation infra; only attempt if Day 28 finishes ahead |
| 2026-05-13 | Use Person Recode (PR) deduped to households when DHS bundles PR not HR | Asset variables are household-level; dedup by (hv001, hv002) recovers HR rows. Avoids re-download. |
| 2026-05-13 | Collapse DHS categorical asset variables (water/toilet/floor/wall/roof) to single binary "improved/finished" flags | Sidesteps country-specific ordinal-coding inconsistencies across 23 countries. Slight signal loss vs Yeh's 1–5 ordinal scoring. |

## Risks currently in play

- **School GPU server access** — RustDesk in exists but the machine was powered off. Friend asked to power it on. Blocking: bulk EE extraction, model training. **Highest-priority risk.**
- **Bulk EE extraction not yet launched** — design called for this Day 2 as a 24–48h background job. Every additional day of delay compresses Week 2.
- **22 of 23 countries' DHS data not downloaded** — purely web work, can be done from the laptop, just hasn't been started.
- **Zero tests in `tests/`** — `dhs.py` and `earth_engine.py` validated only by one manual Kenya run.
- **`README.md` quickstart references scripts that don't exist** — fixed; will need updating again as scripts land.

## What's actually built (snapshot)

- ✅ `src/poverty_cnn/data/dhs.py` — DHS loader + pooled PCA wealth index
- ✅ `src/poverty_cnn/data/earth_engine.py` — 8-channel Landsat+nightlights extraction
- ⬜ `src/poverty_cnn/data/dataset.py` — PyTorch Dataset (next)
- ⬜ `src/poverty_cnn/data/splits.py` — 5-fold cross-country split (next)
- ⬜ `src/poverty_cnn/models/` — empty (Week 2)
- ⬜ `src/poverty_cnn/training/` — empty (Week 2)
- ⬜ `src/poverty_cnn/eval/` — empty (Week 3)
- ⬜ `src/poverty_cnn/viz/` — empty (Week 3)
- ⬜ `scripts/`, `tests/`, `notebooks/` — directories exist, all empty

## Future work (parking lot — not for this 3-week window)

- **OOD generalization test on excluded countries.** Once the 23-country model is trained, request DHS access for additional sub-Saharan countries Yeh excluded (Botswana, Namibia, Madagascar, Liberia, etc.) and run inference-only — no retraining. Report per-country r² on these as a "does the model travel?" test. South Africa is especially interesting because it's upper-middle-income and would test whether the model overfits to low-income visual features.
- Self-supervised satellite-image pretraining (replace ImageNet pretraining with SimCLR / MAE on unlabeled Landsat).
- Building-footprints integration (Microsoft Open Buildings or Google Open Buildings) as additional input.
- Vision Transformer architecture replacing ResNet-18.
- Multi-modal model fusing satellite + Open Street Map road network data.
- Pixel-level wealth maps (the Yeh 2020 24-h imagery pipeline).
