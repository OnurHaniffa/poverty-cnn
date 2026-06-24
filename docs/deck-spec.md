# Final Deck — Page-by-Page Spec

Template: the engineering-blueprint style (graph-paper bg, gears, blueprint node-lines, cyan
chevrons, teal-blue headlines, green number-circles, dark-teal emphasis boxes). Spine **A**
("confidently wrong about the poorest") with **C** ("ranking → targeting") as the throughline.
~25 main slides + a backup appendix. Build order: one page at a time, render + approve each.

Legend — **Type** = which template slide-kind it reuses · **Visual** = the image/figure on it ·
**Beat** = its job in the story.

---

## ACT 0 — FRAME

### S1 · TITLE  *(Type: title slide)*
- **Title:** Predicting Village Wealth from Space
- **Sub:** A fairness & uncertainty audit of satellite poverty mapping · 23 countries
- **Footer:** Onur Haniffa · Advisor: Dr. Seda Nilgün Dumlu · ML/DL Internship 2026
- **Visual:** one clean true-colour satellite tile (replaces the 3D-printer illustration entirely)
- **Beat:** first impression; sets the "from space" theme.

### S2 · WHAT THIS IS — AND ISN'T  *(Type: 3-box / content)*
- **Title:** Three things to know up front
- **Content:** **IS** → a faithful replication + the fairness/uncertainty audit Yeh (2020) didn't run. **ISN'T** → a new architecture; a deploy-ready targeting tool; a criticism of the original.
- **Visual:** none / subtle motif
- **Beat:** disarms the obvious pushback in 30 seconds.

---

## ACT 1 — THE PROMISE

### S3 · SECTION "01 · THE PROMISE"  *(Type: section divider, big "01")*
- **Sub:** Can a free satellite image replace an expensive survey?
- **Visual:** full satellite tile / Earth (replaces the stock photo)

### S4 · THE PROBLEM  *(Type: 2-stat)*
- **Title:** The Problem
- **Stat 1:** **1.8B** — people below the poverty line; the data to find them is scarce, costly, years out of date
- **Stat 2:** **~5 yrs** — typical gap between household surveys
- **Sub-head + bullets ("Why satellites?"):** image everywhere, free & repeatedly · can a CNN read wealth from it? · if yes, a scalable complement to surveys — *and these maps already steer real aid*
- **Beat:** the need + the opportunity (+ the stakes hook).

### S5 · THE IDEA — pipeline  *(Type: 5-step process + emphasis box)*
- **Title:** THE IDEA — one model, one number
- **5 steps:** ① a village's GPS pulls a satellite tile → ② 8 bands (colour, infrared, thermal, night-lights) → ③ a ResNet-18 reads all 8 → ④ it outputs ONE number: predicted wealth → ⑤ trained against DHS survey "ground truth"
- **Box "Why it can work":** wealth leaves a visible footprint (roofs, roads, lights) · imagery is free & global · one pooled model serves 23 countries
- **Beat:** the whole system in one picture.

---

## ACT 2 — THE INGREDIENTS

### S6 · SECTION "02 · THE INGREDIENTS"  *(section divider)*
- **Sub:** Where the wealth number actually comes from.

### S7 · THE ANSWER KEY (DHS)  *(content, 2 sub-headers)*
- **Title:** The answer key: DHS surveys
- **What it gives:** gold-standard household surveys · a cluster ≈ a village (~25 households) · a GPS for every cluster
- **The deliberate twist:** DHS fuzzes each GPS up to 2 km (urban) / 5 km (rural) for privacy → *that's why our tile is a wide 6.7 km box* (big enough to still contain the true village)
- **Visual:** cluster-map of the 23 countries
- **Beat:** the ground truth + why the wide tile. Analogy: the answer key we're graded against.

### S7A · THE DATA IS CLEAN — AND WE CAN PROVE IT (EDA · cleaning)  *(content + figure)*
- **Title:** Before modelling: cleaning the data
- **Content (the pipeline, as 4 steps):** ① DHS special codes (96–99 "don't know") → NaN, imputed by within-country mean · ② drop households missing >30% of asset fields · ③ drop null-island / bad-GPS clusters · ④ satellite tiles: per-channel mean-fill any cloud/NaN pixels.
- **Callout:** "How clean? **Mean 0.66% missing pixels per tile; 95% of tiles under 5%; worst case 19%.** The imagery is essentially complete."
- **Visual:** `eda/missing_data.png` (tile-missingness histogram)
- **Beat:** answers "did you check your data?" before she asks. This is the rigor handshake.

### S7B · THE DATA AT A GLANCE (EDA · distributions)  *(content + 2 figures)*
- **Title:** The data at a glance
- **Content:** wealth spans the full range in every country — **DRC poorest, Ghana richest** — with heavy overlap, so this is *not* a trivial country-ID task. And the split that drives the whole back half: **urban villages sit far above rural**, with rural tightly bunched near the bottom.
- **Visual:** `eda/box_wealth_by_country.png` (23-country box plot) + `eda/box_wealth_urban_rural.png`
- **Beat:** shows the target's spread and *secretly plants the urban/rural fairness theme* (callback in S18).

### S8 · THERE IS NO "WEALTH" COLUMN (PCA)  *(content + figure)*
- **Title:** There is no "wealth" column
- **Content:** DHS records asset checkboxes (electricity? TV? fridge? car? finished floor?). **PCA** finds the single mix of assets that best explains who owns *more* — that axis (PC1) *is* wealth. Average it to the village = our target.
- **Visual:** PCA loadings bar chart (`eda/02_pca_loadings.png`)
- **Beat:** how 15 yes/no answers become one number. Analogy: a credit-score formula discovered from data. (Pre-empt "only 28%?": 4× the random floor; validity is in the all-positive loadings.)

### S9 · EIGHT WAYS TO SEE A VILLAGE (the money shot)  *(content / big imagery)*
- **Title:** Eight ways to see a village
- **Content:** 8 bands = 8 "senses": colour · near-infrared (vegetation & rooftops) · SWIR · thermal (heat) · **night-lights (electrification)**. Each band is a **median over a full year, across a 3-year window** → one clean, cloud-free image. The CNN reads all 8 together → one number.
- **Visual:** the 8-band grid + composite, Nairobi (urban) vs Turkana (rural); an "8 → 1" arrow motif
- **Beat:** the input, vividly; secretly plants the urban/rural theme. (the 8→1 arrow + "365 days → 1 image, over 3 years".)

### S10 · THE MODEL, IN 20 SECONDS  *(content, short)*
- **Title:** The model, in 20 seconds
- **Content:** a ResNet-18 (standard CNN) · input rebuilt for 8 channels (not 3) · output is one number (not categories) · trained from scratch.
- **Punch line:** "The science is in the data and the honest evaluation — not the architecture."
- **Beat:** de-emphasise the model; tee up the rigor.

---

## ACT 3 — HOW WE TEST HONESTLY

### S11 · SECTION "03 · DOES IT WORK?"  *(section divider)*
- **Sub:** First — the honest test.

### S12 · TESTING ON COUNTRIES IT'S NEVER SEEN  *(content — the professor's favourite)*
- **Title:** The honest test: cross-country validation
- **Content:** random split → the model can memorise a country (flattering but fake). We hold out **whole countries** — train on ~18, test on 5 it has never seen, rotate so each is tested once. *Every number we report is on unseen countries.*
- **Callout box:** "We measured what cheating would have bought — a naïve random split scores **0.73**; the honest country-blocked number is **0.61** (+20% inflation we refuse)."
- **Visual:** the CV-ladder (random < within-country < leave-country-out < temporal < OOD)
- **Beat:** the rigor centrepiece. Analogy: an exam with questions you never studied.

### S13 · HOW DO WE KNOW IF IT'S GOOD? (metrics)  *(content + worked example)*
- **Title:** Three metrics, three questions
- **Content:** **r²** (does it capture the spread?) · **MAE** (how many points off?) · **Spearman** (did it get the *order* right?). For finding the poor, **ranking is what matters**. (We also report Pearson r, worst-group r, targeting recall.)
- **Visual:** a 5-village worked-example table (TO BUILD)
- **Beat:** metric literacy; makes "ranking" the throughline.

---

## ACT 4 — IT WORKS (the "ranking" half)

### S14 · DOES IT WORK? YES — AT BENCHMARK LEVEL  *(2-stat)*
- **Title:** Does it work? Yes — at benchmark level
- **Stat 1:** **r 0.78** — matches the WILDS PovertyMap benchmark (its own metric)
- **Stat 2:** **0.55** — worst-group r, beats their 0.45
- **Bullets:** pooled r² 0.61 on unseen countries · over 2× a naïve floor · *honest note: this is a metric reframe — the r² gap to Yeh is the data, not the method (a pretrained model doesn't help, −0.03)*
- **Visual (optional):** learning-curve + data-scaling 2-panel (no memorising; gap is data)
- **Beat:** it genuinely works.

### S15 · WHAT CARRIES THE SIGNAL? NIGHT-LIGHTS  *(content + bars)*
- **Title:** What carries the signal? Night-lights.
- **Content:** night-lights *alone* ≈ the full 8-channel model. Lights = electrification = the strongest village-wealth signal. Foreshadow: if it's mostly lights, what happens where there are none?
- **Visual:** channel-ablation bars (restyled)
- **Beat:** sets up the limit.

### S16 · IT EVEN WORKS ON UNSEEN COUNTRIES  *(content + bars)*
- **Title:** It even generalises to countries it never saw
- **Content:** the frozen model on **6 brand-new countries** → ranking *transfers* (Spearman 0.65–0.81; **Gabon 0.81 beats its home turf 0.76**). The map is genuinely **good for ranking**.
- **Visual:** OOD per-country Spearman bars (`ood_where_it_breaks`, restyled)
- **Beat:** the peak of the "it works" half — right before the turn.

---

## ACT 5 — BUT IT FAILS THE POOREST (the heart)

### S17 · SECTION "04 · BUT…" (THE TURN)  *(full-bleed image divider)*
- **Big line:** But it fails the people it's meant to find.
- **Visual:** full-bleed *rural* satellite tile (Turkana), darkened
- **Beat:** the pivot. Let it land.

### S18 · IT SERVES CITIES BETTER THAN VILLAGES  *(content + figure)*
- **Title:** It serves cities better than villages
- **Content:** *equal* absolute accuracy (MAE), but it **ranks rural villages worse** (Spearman 0.41 vs 0.54) — in **21 of 23** countries. (Honest aside: partly rural homogeneity, mostly the model.)
- **Visual:** urban-vs-rural figure (`fairness/01_urban_rural`, restyled)
- **Beat:** first fairness crack. Callback Nairobi/Turkana.

### S19 · CONFIDENTLY WRONG ABOUT THE POOREST  *(content + cone figure)*
- **Title:** Confidently wrong about the poorest
- **Content:** the model **regresses to the mean** (slope 0.60) — it predicts the **poorest as richer than they are**, with the *largest* errors exactly there. Used to target the poorest 20%, it catches only ~half — worse for the rural poor.
- **Visual:** regression-to-mean **cone** plot (TO BUILD) + bias-by-decile
- **Beat:** the core failure.

### S20 · CALIBRATION IS NOT EQUITY  *(box-grid — the novelty centrepiece)*
- **Title:** Calibration is not equity
- **Content:** three uncertainty methods (deep ensemble, MC-dropout, heteroscedastic). **None flag the poorest.** Even the *well-calibrated* one gives the poor and the rich **equal** confidence. Why: error = aleatoric + **bias** + variance, and uncertainty is *blind to the bias term*.
- **Visual:** a 2×2 matrix [Calibrated? × Ranks-errors?] (TO BUILD)
- **Beat:** the most novel result. "Forget calibrated UQ — it can't warn you where it's most wrong."

### S21 · WE TRIED TO FIX IT — NOTHING WORKS  *(box-grid)*
- **Title:** We tried to fix it. Nothing works.
- **Content:** loss-reweighting (LDS) — no · Balanced-MSE — no (it just destroys accuracy) · **3× more data** — no (the bias barely moves) · and it **recurs out-of-distribution** (South Africa, the rich extreme, r² −1.7).
- **Visual:** small-multiples or the OOD-break panel
- **Beat:** unfixable — 3 independent legs + 2 regime-replications.

### S22 · WHY? THE LIGHTS ARE DARK  *(content + the killer figure)*
- **Title:** Why? Because the lights are dark.
- **Content:** we **measured** it — night-light intensity is **flat and near-zero across the poorest half** of villages, then climbs steeply. The dominant signal is *absent* for the poor → the satellite literally cannot separate the extreme poor.
- **Visual:** night-light-by-wealth-decile hockey-stick (`nightlight_by_wealth`, restyled)
- **Beat:** the answer. Converts the thesis from *argued* to *shown*.

---

## ACT 6 — SO WHAT

### S23 · SECTION "05 · SO WHAT?"  *(section divider)*
- **Sub:** What it means for actually using these maps.

### S24 · THE COST OF TARGETING WITH IT  *(2-stat / content)*
- **Title:** If you used it to target aid…
- **Content:** target the poorest 20% by the model → you'd **miss ~half** the truly-neediest villages; aid skews toward the less-poor; the **rural poor** are missed most.
- **Visual:** targeting/recall figure (TO BUILD from existing preds)
- **Beat:** the human stakes.

### S25 · THE ONE THING TO REMEMBER  *(big-statement slide)*
- **Big line:** Satellite poverty maps are great for **ranking regions** — and dangerous for **targeting the poorest**. And standard uncertainty can't warn you when they're wrong about the neediest.
- **Beat:** the single takeaway.

### S26 · WHAT'S NEW HERE (contributions)  *(conclusion grid)*
- **Foundation:** faithful replication; matches the WILDS benchmark; night-light dominance.
- **Modest novelty:** 23-country audit (vs Aiken's 10); decile-resolved bias; temporal + OOD external validity.
- **Most novel:** the equity-framed uncertainty negative ("calibration ≠ equity").
- **Honest:** a rigorous audit, not a method — no new SOTA claimed.

### S27 · THANK YOU / QUESTIONS  *(thanks slide)*
- "Thank you · Questions?" + a memorised back-pocket: 0.78 / 0.55 / poorest-bias +0.59 / S.Africa r² −1.71 / AURG ≈ 0.

### S28 · REFERENCES  *(references slide)*
- Yeh 2020 (Nat. Comms) · Aiken 2022/2023 · Koh 2021 (WILDS) · Kattenborn 2022 · Yang 2021 (LDS) · Ren 2022 (Balanced-MSE).

---

## BACKUP / APPENDIX (hidden, flip to when asked)
One slide each, built from `methodology-defense.md`:
- **B1** Why leave-country-out, not nested CV? (documented limitation + the 0.64→0.52 evidence)
- **B2** Why from-scratch, not pretrained? (tested — pretraining doesn't help, −0.03; gap is data)
- **B3** How did you handle imbalanced data? (LDS + Balanced-MSE, both fail = signal limit)
- **B4** How do you know there's no leakage? (country-disjoint folds + train-only norm + measured +20% inflation + residual Moran's I)
- **B5** Why these metrics / why Spearman? (ranking = targeting; Spearman avoided only for *uncertainty*, where we used AUSE)
- **B6** Why not Sentinel-2 / a SOTA backbone? (data-limited; pretraining didn't help → fancier won't; future work)
- **B7** Is the poorest-bias significant? (bootstrap CIs; OOD break CI non-overlapping)

**EDA deep-dive (flip to for "why didn't you check X?")**
- **B8** Channel distributions — the 8 satellite bands standardised; **night-lights is the most skewed** (long bright-city tail), the rest near-symmetric (`eda/box_band_distributions.png`).
- **B9** Channel correlation — RGB tightly coupled; **night-lights nearly independent** of the optical bands → each carries distinct signal, justifying all 8 (`eda/band_correlation.png`).
- **B10** The wealth index, validated — PCA scree (PC1 captures the dominant axis) + all-positive loadings, asset prevalence (`eda/01_pca_scree.png`, `eda/02_pca_loadings.png`, `eda/03_asset_prevalence.png`).
- **B11** Sample & coverage — clusters per country, households per cluster, geographic spread (`eda/08`, `eda/09`, `eda/10`).

---

## Figures to build/restyle (into the teal/cyan/navy palette)
1. Metrics worked-example table (S13) — new
2. Regression-to-mean **cone** (S19) — new
3. UQ **2×2** calibrated×ranks-errors (S20) — new
4. OOD ranking bars (S16) + OOD-break panel (S21) — restyle existing
5. Night-light hockey-stick (S22) — restyle existing
6. Channel-ablation bars (S15), urban/rural (S18), PCA loadings (S8), targeting (S24) — restyle existing
7. Clean true-colour satellite tiles (S1, S3, S9, S17) — render caption-free
8. **EDA — DONE** (scripts/50_eda.py, in palette): country box plot, urban/rural box plot, missingness, band distributions, channel correlation → S7A, S7B, B8, B9.
