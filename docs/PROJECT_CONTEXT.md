# Project Context — full narrative

A complete catch-up document for a new collaborator or a fresh working session.
For the authoritative plan see `docs/design.md`.

Last updated: 2026-05-14 (end of Day 5 work).

---

## 1. Who and why

- **Onur Haniffa** — undergraduate at Acıbadem Mehmet Ali Aydınlar University, Turkey. Aspiring data engineer. Strong practical engineer — previously shipped "ivory-ai", a full FastAPI/PostgreSQL/OpenAI SaaS product.
- **Advisor:** Dr. Seda Nilgün Dumlu.
- **Context:** ML/DL internship, Spring 2026. The first internship project was a "replicate + critique + extend" study of Khanam & Foo (2021) diabetes prediction — it exposed that the paper's headline 88.6% NN accuracy was unreproducible and that weighted-average reporting hid a Class 1 recall of 0.549. The advisor then asked for a CNN / image-recognition project "for practice."
- **What the user wanted:** something genuinely impressive on a CV, real-world impact, public data, the same "replicate + expose hidden flaw + improve" arc as the diabetes paper. Explicitly wanted to "overdeliver."

## 2. How we landed on this project

We brainstormed extensively. Candidates considered and rejected:

- **PlantVillage** (plant disease CNN) — strong "replicate + bias audit" story but a saturated, bootcamp-tier topic. Rejected as not impressive enough.
- **HAM10000 / PCam / medical imaging** — every ML undergrad does medical imaging; too saturated.
- **EuroSAT land use** — econ-adjacent but indirect.
- **Raw Jean 2016 / Yeh 2020 replication from scratch** — considered, but the original codebases are dead (TF 1.15, 500 GB, GDAL hell). High risk for a 3-week window.

**Chosen:** Replicate **Yeh et al. 2020** (*Nature Communications*) via a clean PyTorch 2.x reimplementation, using the same public data (DHS surveys + Landsat). Top-tier venue, real-world development-economics impact, genuine data-engineering signal (multi-source geospatial pipeline), and a clean fairness-audit critique angle. WILDS PovertyMap exists as a documented fallback if the raw Earth Engine pipeline collapses.

## 3. The papers

- **Jean et al. 2016, *Science*** — "Combining satellite imagery and machine learning to predict poverty." 5 countries, multi-step transfer learning (ImageNet → fine-tune on nighttime lights → ridge regression). Asset-wealth r² 0.55–0.75.
- **Yeh et al. 2020, *Nature Communications*** — direct successor. 19,669 villages, 23 countries, end-to-end ResNet-18 on 8-channel Landsat+nightlights. Mean cross-country r² = 0.70. Code: `github.com/sustainlab-group/africa_poverty` (TF 1.15). **This is the primary replication target.**
- **Aiken, Rolf, Blumenstock 2023, IJCAI** — "Fairness and representation in satellite-based poverty maps." Audited urban/rural disparities in 10 countries. **The critique we extend to all 23.**

## 4. The four contributions

1. **Replicate** Yeh 2020 (mean cross-country r² = 0.70) with a modern PyTorch 2.x pipeline.
2. **Fairness audit** — per-country + urban/rural across all 23 countries, with stratified targeting-accuracy analysis (extends Aiken 2023's 10 countries).
3. **Uncertainty-aware fairness extension** (novel) — MC-dropout per-cluster uncertainty; show uncertainty is unequally distributed; propose an uncertainty-aware aid-allocation rule.
4. **Temporal fairness drift** (novel) — train on Period 1+2 (2009–2014), test on Period 3 (2015–2017), see whether the urban/rural gap widens over time.

Plus a stretch goal: bias-amplification simulation. Full detail in `docs/design.md`.

## 5. Timeline (from docs/design.md)

- **Week 1 (May 8–15):** data + infrastructure. Hard checkpoint May 15: end-to-end pipeline working.
- **Week 2 (May 16–22):** replication + main result. Hard checkpoint May 22: replication numbers in hand.
- **Week 3 (May 23–30):** fairness audit, novel extensions, write-up.

**As of Day 5 (May 13) we are roughly a day AHEAD of schedule** — all approvals are in and the data pipeline modules are written and tested, which the spec scheduled for later in Week 1.

## 6. What's been done — day by day

### Day 1–3 (May 8–11): bootstrap
- Created `poverty-cnn` repo at `/Users/onurmohamedhaniffa/poverty-cnn/`, pushed to GitHub (public): `github.com/OnurHaniffa/poverty-cnn`.
- Wrote `environment.yml`, `README.md`, `.gitignore`, `pyproject.toml`, `LICENSE` (MIT), directory skeleton.
- Installed miniforge via Homebrew, created the `poverty-cnn` conda env.
- Copied the design spec into `docs/design.md`, created `docs/tasks.md`.
- Submitted DHS data-access application for all 23 countries (Survey + GPS).
- Registered Google Earth Engine (noncommercial tier, project `storied-chimera-491721-i4`).

### Day 4 (May 12): approvals + auth
- DHS approved **both Survey and GPS** for all 23 countries. Auth letter saved to `private/DHS_AuthLetter_2026-05-11.pdf`.
- Earth Engine approved.
- Fixed the conda env: numpy<2 pin, manually pip-installed earthengine-api / wilds / grad-cam (mamba's pip block had failed silently). Smoke test passed — MPS available, 8-channel tensor works on the Mac GPU.

### Day 5 (May 13): data pipeline built + tested
- Wrote `src/poverty_cnn/data/earth_engine.py` — full Yeh 2020 protocol 8-channel extraction. Smoke-tested with Nairobi 2017.
- Wrote `src/poverty_cnn/data/dhs.py` — DHS loader + pooled PCA wealth index.
- Authenticated Earth Engine on the laptop, verified with a real Landsat query.
- Downloaded Kenya 2014 DHS data (came as Person Recode + GPS bundle).
- **End-to-end test on Kenya 2014:**
  - 36,430 households → deduped → 1,594 clusters
  - Pooled PCA: PC1 explains 26% of variance, all loadings positive (finished floor +0.40, electricity +0.39, TV +0.38 strongest)
  - Wealth + GPS joined: 1,585 clusters; urban mean +0.61, rural mean −0.38 — a **0.99σ urban/rural gap** (the disparity the fairness audit will study)
  - Two satellite tiles downloaded: Nairobi (urban, wealth +3.14, nighttime light = 13) vs Turkana rural (wealth −1.41, nighttime light = 0). The signal is unmistakable.
- Committed (`79306e3`) and pushed.

### Day 6–7 (May 20–21): infra unblocked, all data collected, extraction running
- **Remote GPU access solved.** Lab PC is behind NAT with a dynamic IP, so direct SSH failed. Diagnosed it (Ubuntu 24.04, 2× Quadro P5000 16 GB), then set up **Tailscale**. Twist: the PC was already on a friend's tailnet (`1boraguvendiren@`), so the friend **node-shared** `sim-vs` to `onurhturkey@`. Now **`ssh sim` works from anywhere** (alias in `~/.ssh/config`, key auth — the key install needed `ssh-copy-id` because copy-paste mangled the long key line). RustDesk is the break-glass backup.
- **All 23 countries' DHS data downloaded + verified** (Survey HR + GPS) into `data/raw/dhs/<CC>/`. ~13,634 clusters. Senegal is a Continuous DHS (`SN_..CONTINUOUSDHS`); the rest standard. (SL = Sierra Leone, not Senegal — easy mix-up.)
- **Bug found + fixed:** `cluster_image` now `.toFloat()`s the 8-band stack. EE batch GeoTIFF export rejects mixed Float64/Float32 bands — only surfaced on the `toDrive` bulk path; the Day-5 direct-download tolerated it. Caught by validating on `--limit 3` before firing 1,585.
- **Kenya extraction complete** (1,594 tiles in Drive); **full 22-country extraction launched** from the Mac (`caffeinate` background) → Drive `poverty_cnn_data` (now 100 GB Google One).
- **OOD test countries approved + banked** (don't extract til core trained): South Africa, Namibia, Gabon, Eswatini, Madagascar, Niger, Liberia.
- **Scope decision:** lock the core 23 single-round dataset; OOD test (project onto the *frozen* 23-country PCA axis — never re-fit) and multiple-rounds temporal analysis are Week-3 stretches.
- **PC env:** package CDNs are painfully slow from this region (~200 KB/s); Miniconda + conda env build run overnight.
- Built `scripts/03_download_imagery.py` (resumable, quota-aware bulk submitter) and `scripts/visualize_tile.py`; refactored `dhs.py` (semantic names, hv216 special-code fix, sign-flip pinned to `has_electricity`). Committed (`3dd0a46`, `1f11465`) + pushed.

## 7. Access / credentials status

| Resource | Status | Notes |
|---|---|---|
| GitHub repo | ✅ Live, public | `github.com/OnurHaniffa/poverty-cnn` |
| Conda env | ✅ Working | `/opt/homebrew/Caskroom/miniforge/base/envs/poverty-cnn/` |
| DHS data access | ✅ Approved, all 23 countries | login: onurhturkey@gmail.com at dhsprogram.com |
| Earth Engine | ✅ Authenticated | project `storied-chimera-491721-i4`, noncommercial tier |
| School GPU server | ✅ Reachable via `ssh sim` | Tailscale (node-shared from friend's tailnet), IP `100.101.91.62`. 2× Quadro P5000 16 GB, Ubuntu 24.04. Must be powered on; RustDesk is the backup. |

## 8. Key decisions and why

| Decision | Rationale |
|---|---|
| Replicate Yeh 2020, not Jean 2016 or PlantVillage | Top venue, modern end-to-end pipeline, public data, real impact, fits "aspiring data engineer" career path |
| 5-fold cross-country CV with Yeh's Supp Table S2 fold assignment | Apples-to-apples comparison with the paper; standard CV; not 7/8-fold (too few test countries per fold), not exhaustive |
| Use exactly Yeh's 23 countries — no adding South Africa etc. | Adding countries changes the pooled PCA → wealth index no longer comparable to Yeh's. OOD test on excluded countries is logged as future work. |
| Added uncertainty + temporal extensions as novel contributions | Turns "solid undergraduate replication" into "small but real research contribution" — neither has been done on the full 23-country scale |
| Bias-amplification simulation = stretch only | Highest novelty but needs simulation infra; only if Week 2 finishes ahead |
| Use Kenya 2014 (not 2022) as the sandbox country | Yeh used the 2009–2017 data window; 2022 is DHS-8 with different variable codes and breaks apples-to-apples |
| Use the Person Recode file deduped to households | DHS bundled PR instead of HR; asset variables are household-level anyway, so dedup recovers the HR rows. Saves a re-download. |

## 9. Gotchas / lessons

- Conda not shell-initialized — use full paths to env binaries.
- mamba's `pip:` section in environment.yml failed silently — pip-install those packages manually.
- numpy 2.x breaks PyTorch 2.2.x — pinned numpy<2.
- DHS sometimes bundles Person Recode (PR) instead of Household Recode (HR) — `load_dhs_pr_as_hr()` handles it.
- Earth Engine `getDownloadURL` can take minutes — 600s timeout, streamed download.
- EE tiles come back 225×224 / 226×224 — Dataset class must center-crop to exactly 224×224.

## 10. What's next (Days 6–7, finishing Week 1)

1. **School server SSH + VS Code Remote-SSH** — once the machine is powered on. RustDesk in, set up sshd, add public key, configure `~/.ssh/config`.
2. **Download DHS data for the remaining 22 countries** — HR + GE per country into `data/raw/dhs/<CC>/`.
3. **Bulk Earth Engine extraction script** — loop all 23 countries' clusters → `export_cluster_to_drive()` → Google Drive → rsync to school server. 24–48h background job.
4. **`src/poverty_cnn/data/dataset.py`** — PyTorch Dataset wrapping (8-channel tile, wealth) pairs; center-crop to 224×224.
5. **`src/poverty_cnn/data/splits.py`** — 5-fold cross-country split from Yeh Supp Table S2.
6. **WILDS parity check** — verify our raw-pipeline output matches the WILDS PovertyMap subset for ≥3 overlap countries.

Then Week 2 = model + training on the school GPU server.

## 11. Repo map

```
poverty-cnn/
├── README.md                 public-facing project overview
├── environment.yml           conda env spec (numpy<2 pinned)
├── pyproject.toml            editable-install package metadata
├── LICENSE                   MIT
├── .env                      GEE_PROJECT_ID (gitignored)
├── docs/
│   ├── design.md             the 772-line authoritative spec
│   ├── tasks.md              live progress tracker
│   ├── dhs-application.md    DHS form text for the user's records
│   └── PROJECT_CONTEXT.md    this file
├── src/poverty_cnn/
│   ├── data/
│   │   ├── dhs.py            ✅ DHS loader + pooled PCA wealth index
│   │   └── earth_engine.py   ✅ 8-channel Landsat+NL extraction
│   ├── models/               (empty — Week 2)
│   ├── training/             (empty — Week 2)
│   ├── eval/                 (empty — Week 3)
│   └── viz/                  (empty)
├── data/                     gitignored — raw + processed data
├── results/                  gitignored — checkpoints, predictions, figures
└── private/                  gitignored — auth letters, personal docs
```
