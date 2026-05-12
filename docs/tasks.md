# Project tasks

Live tracking of where the project stands. Updated as work progresses.

See [`design.md`](design.md) for the full design and timeline.

## Status

| Phase | Status | Notes |
|---|---|---|
| Day 1 — Setup | 🟡 In progress | Started 2026-05-09 |
| Week 1 — Data and infrastructure (May 8–15) | ⏳ Pending | Hard checkpoint May 15 |
| Week 2 — Replication and main result (May 16–22) | ⏳ Pending | Hard checkpoint May 22 |
| Week 3 — Fairness audit + extensions + write-up (May 23–30) | ⏳ Pending | |

## Day 1 — bootstrap (today)

- [x] Create `poverty-cnn/` repo skeleton
- [x] Write `environment.yml`, `README.md`, `.gitignore`, `pyproject.toml`, `LICENSE`
- [x] Copy spec to `docs/design.md`
- [ ] Initialize git, first commit
- [ ] Create GitHub repo (public), push
- [ ] Register for DHS data — submit project description, await 1–3 day approval
- [ ] Register for Google Earth Engine — await ~1 day approval
- [ ] Create local conda environment, run smoke test
- [ ] Configure VS Code Remote-SSH to school server (awaits server credentials)

## Week 1 schedule (May 8–15)

| Day | Task |
|---|---|
| Day 1 (Fri May 8) | Conda env. SSH config. DHS + EE registrations. (Buffer: original Day 1 work spread to today since real start is May 9.) |
| Day 2 (Sat May 9) | **Today.** Repo skeleton, registrations, env setup. |
| Day 3 (Sun May 10) | Read `africa_poverty` repo end-to-end. Read `jmather625/predicting-poverty-replication` end-to-end. Read Aiken 2023 IJCAI paper. Launch background EE download script. |
| Day 4 (Mon May 11) | Pull DHS GPS clusters for all 23 countries. Compute pooled asset wealth index by PCA. Verify against WILDS. |
| Day 5 (Tue May 12) | Continue EE download. Build PyTorch `Dataset` class. Write data loading tests. |
| Day 6 (Wed May 13) | Match Yeh 2020 Supplementary Table S2 fold assignment. Write CV split logic. Start `train.py` skeleton. |
| Day 7 (Thu May 14) | Verify all 23 countries' tiles downloaded. Verify pixel statistics match WILDS for 3 overlap countries. Build modified ResNet-18. Forward-pass smoke test. |
| Day 8 (Fri May 15) | **Hard checkpoint.** End-to-end pipeline working. If yes, commit and tag `pipeline-v1`. If no, fall back to WILDS. |

## Week 2 schedule (May 16–22)

See `docs/design.md` §11 for the full Week 2 day-by-day.

## Week 3 schedule (May 23–30)

See `docs/design.md` §11 for the full Week 3 day-by-day.

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | Picked Yeh 2020 (Nature Comms) replication over PlantVillage / HAM10000 / PCam | Top venue, real-world impact, fits aspiring-data-engineer career path |
| 2026-05-09 | 5-fold cross-country protocol matching Yeh's Supp Table S2 | Apples-to-apples comparison; standard CV; not 7/8/exhaustive |
| 2026-05-09 | Added uncertainty-aware fairness extension as novel contribution | Real research novelty; cheap (no extra training) |
| 2026-05-09 | Added temporal fairness drift extension as novel contribution | Real research novelty; ~5–10 GPU-h additional |
| 2026-05-09 | Marked bias-amplification simulation as Week 3 stretch | Highest novelty but requires simulation infra; only attempt if Day 28 finishes ahead |

## Risks currently in play

- DHS approval delay (1–3 days) — submitted but awaiting
- Earth Engine approval delay (~1 day) — submitted but awaiting
- School server SSH access not yet configured
- First-CNN debugging cycle ahead — expect 1–2 days of model-not-converging issues

## Future work (parking lot — not for this 3-week window)

- **OOD generalization test on excluded countries.** Once the 23-country model is trained, request DHS access for additional sub-Saharan countries Yeh excluded (Botswana, Namibia, Madagascar, Liberia, etc.) and run inference-only — no retraining. Report per-country r² on these as a "does the model travel?" test. South Africa is especially interesting because it's upper-middle-income and would test whether the model overfits to low-income visual features.
- Self-supervised satellite-image pretraining (replace ImageNet pretraining with SimCLR / MAE on unlabeled Landsat).
- Building-footprints integration (Microsoft Open Buildings or Google Open Buildings) as additional input.
- Vision Transformer architecture replacing ResNet-18.
- Multi-modal model fusing satellite + Open Street Map road network data.
- Pixel-level wealth maps (the Yeh 2020 24-h imagery pipeline).
