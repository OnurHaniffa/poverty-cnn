# Presentation prep — discussion points

A running, accumulating list of the **methodological and logical questions** that have come up while building the project. Each entry captures something a careful audience member or advisor might ask, plus the answer in a form you can deliver in a slide or in conversation.

These are *not* generic Q&A. Every question here is one Onur actually asked while working through the code — meaning the answers reflect real decision points and real critiques, not boilerplate.

## How to use this file

- **Before the talk:** read top to bottom, mentally rehearse the answers.
- **During the talk:** have these answers warm if your advisor or the audience pushes on any of them.
- **For the final report:** most of these can become "Methods choices and rationale" subsections.
- **Going forward:** keep adding new questions as we work through `earth_engine.py`, the dataset module, training, and the fairness audit. The file accumulates over the whole 3-week sprint.

---

## 1. Replication strategy

### What kind of "replication" is this?

**The question:** Does "replicating Yeh 2020" mean we have to reproduce every single implementation choice exactly?

**Why it matters:** Audience may assume "replication" = bit-exact reproduction. It isn't.

**Answer:**
- Replication in applied ML / applied stats sits on a spectrum: bit-exact → computational → **methodological** → conceptual.
- We're doing **methodological replication**: re-implementing the *method* (as described in the paper), independently, on the same data, in modern tooling.
- Methodological replications **are expected to have small, documented divergences** from the original. The 0.05–0.15 r² gap that genuine replications often show is *literally because* of small implementation details the original authors didn't publish.
- Our job isn't to reproduce every micro-choice. Our job is to implement the method faithfully enough that the scientific claim is testable on the same data — and to **document every divergence we introduce**.

### Where do we knowingly diverge from Yeh 2020?

**The question:** What modernization choices have we made that differ from Yeh's implementation?

**Why it matters:** Demonstrates transparency. Each divergence is documented and bounded.

**Answer (a partial list — will grow):**

| Yeh 2020 | Our choice | Why |
|---|---|---|
| TensorFlow 1.15 | PyTorch 2.x | TF 1.15 is deprecated, can't run |
| Landsat Collection 1 | Landsat Collection 2 | C1 was decommissioned by USGS in 2022 |
| Ordinal 1–5 asset coding | Binary improved/finished indicators | Cross-country code-set robustness (see §2) |
| ResNet-18 v2 with pre-activation | torchvision's standard ResNet-18 | What torchvision exposes |
| Adam | Adam (likely AdamW later) | Modern default |
| Older numpy/sklearn | Current versions | Can't avoid library drift |

**Every single one** of these is a methodological divergence. We acknowledge them all and bound the expected impact on the final r².

### Why the binary collapse of categorical assets is defensible

**The question:** Yeh scored floor/wall/roof on a 1–5 ordinal scale. We collapse them to binary "improved/finished". Does that break the replication?

**Why it matters:** This is one of the largest visible methodological divergences. Worth a clear, prepared answer.

**Answer:**
- The DHS categorical codes (hv201, hv205, hv213, hv214, hv215) take values like 11 (piped water), 21 (tubewell), 31 (parquet floor), 34 (cement), etc. Yeh assigned each a 1–5 quality score.
- We instead collapse to binary using WHO/UNICEF "improved" and DHS "30-series finished" code sets.
- **Why this is *arguably better* for our 23-country pooled setting, not just easier:**
  - The 1–5 ordinal mapping drifts subtly between countries and DHS rounds. When you pool 23 countries, you stack 23 slightly-mismatched ordinal scales — adding cross-country noise.
  - The binary mapping is **stable across all countries and rounds** — "is this floor in the standard finished-material code set?" has the same answer everywhere.
  - For cross-country pooled PCA, the cross-country consistency gain likely outweighs the within-country resolution loss.
  - **Filmer & Pritchett 2001** (the foundational asset-index paper, ~4000 citations) uses binary asset coding. DHS itself publishes a binary-derived wealth index (`hv270`).
- **Expected cost:** PC1 explained variance drops by ~3–5 percentage points (we saw 26% on Kenya; ordinal would probably hit ~30%). Per-cluster wealth values change by <5%. Well inside the 0.05–0.15 r² divergence budget.
- **Why it doesn't undermine our scientific claims:** three of our four contributions (fairness audit, uncertainty extension, temporal drift) depend on **rank-order differences between subgroups**, not absolute wealth values. Those are very rank-stable across binary vs ordinal coding.

### Should we make our methodological divergences "somewhere clever" instead?

**The question:** Could we get away with not diverging by making this choice elsewhere?

**Why it matters:** Reflects sharp thinking. The answer reframes the whole question.

**Answer:** We're already diverging in many places (see the table above). The wealth-index choice is **one item on a long list of necessary modernization compromises**. The right framing isn't "pick one place to diverge cleverly"; it's "minimize divergences where possible, document them all, bound their expected impact." We're doing all three.

---

## 2. The wealth index

### Why pool households across all 23 countries before running PCA?

**The question:** Why not run PCA separately per country?

**Why it matters:** This is the single most important choice in the wealth-index recipe.

**Answer:**
- The wealth index has to be **comparable across countries** for fairness comparisons to make sense.
- If you ran PCA per country, you'd get 23 separate PC1 axes. A "+1.0" wealth score in Kenya would mean "richer than average Kenyan"; a "+1.0" in Nigeria would mean "richer than average Nigerian" — totally different things. You couldn't compare across countries.
- By pooling first, you fit **one shared wealth axis** across all 500,000+ households. A "+1.0" anywhere on the axis means the same thing — "one standard deviation richer in assets than the global mean."
- This is also why adding extra countries (South Africa, Botswana) would silently break comparability with Yeh: it would shift the global mean and rescale the axis.

### What does "PC1" actually mean?

**The question:** Is PC1 just the "most important feature"?

**Why it matters:** Common misconception. Getting this right anchors the whole methods explanation.

**Answer:**
- **PC1 is not one of the original features.** It's a **brand-new variable** PCA invents by mixing the original features together with a learned weighted recipe.
- For each household: their PC1 score = sum of (loading × standardized value) across all 15 features.
- The "loadings" (the recipe weights) tell us which original features contribute most to the wealth axis — those are the "feature importance" interpretation. In our data: electricity, finished floor, TV, and improved water/toilet load highest.
- For the wealth axis to align with intuitive wealth, we use a defensive sign-flip (anchored to electricity) to guarantee positive = wealthier.

### What does PC2 mean? Why don't we use it?

**The question:** PCA produces all 15 components. We only keep PC1. What are we throwing away?

**Why it matters:** Shows we understand the full method, not just what we're keeping.

**Answer:**
- PC2 is the **second-strongest pattern** in the data, constrained to be perpendicular (orthogonal) to PC1.
- In DHS asset data, PC2 often captures **"rural-asset mix vs urban-asset mix"** — households at one end own urban-coded assets (TV, fridge, mobile, electricity), households at the other end own rural-coded assets (motorcycle, bicycle, finished roof).
- We discard PC2 because the wealth index is conventionally PC1 only (matches Yeh and Filmer-Pritchett).
- **Bonus interpretation:** because PC2 in this data tends to correlate with urban/rural, it's a sanity check that DHS's URBAN_RURA field is detecting a real pattern in the asset data, not just a labeling convention.

### Why aggregate household scores to cluster level?

**The question:** Why not keep household-level predictions?

**Why it matters:** Audience may not see why the granularity matters.

**Answer:**
- **The CNN sees a place, not a household.** A satellite tile shows roads, rooftops, lights, vegetation across a 6.72 km square — it can't identify individual families.
- Therefore the target has to be at the same spatial granularity as the input: **one wealth number per village**.
- Per-household wealth is noisy (one family might splurge on a TV, another might be passing through). **Averaging 25–30 households cancels random per-household noise** by a factor of √N ≈ 5.
- The averaged "cluster wealth" is precise enough to be a regression target.

### Why is the sign-flip in `pooled_wealth_index` actually critical?

**The question:** Why do we explicitly check the sign of electricity's loading?

**Why it matters:** It's a 5-line defensive snippet that prevents a class of silent corruption. Worth understanding.

**Answer:**
- PCA's component direction is **mathematically arbitrary**. The same PC1 axis could come out with electricity loading +0.39 *or* −0.39 with equal validity. They're the same axis, just numbered backwards.
- Whichever direction PCA happens to produce depends on numerical implementation details (BLAS library version, even).
- Without the sign-fix, half our runs could come out with electricity loading negative → wealth scores flipped → every downstream comparison silently corrupted.
- **The fix:** pick a feature whose direction is unambiguous (electricity = wealth = positive) and force the sign so it loads positive. Both `pc1` and `pca.components_` get flipped together.

---

## 3. The data pipeline

### Are DHS clusters predefined or do we create them?

**The question:** Where do the cluster boundaries come from?

**Why it matters:** Clarifies that the project inherits a sampling design, doesn't invent one.

**Answer:**
- **DHS defines the clusters.** We just inherit them. Cluster IDs, locations, and urban/rural status are fixed by DHS's sampling methodology before any survey-taker knocks on a door.
- DHS uses two-stage cluster sampling. Stage 1: pick clusters (based on national census enumeration areas). Stage 2: within each cluster, randomly select ~25–30 households to interview.
- **Rural cluster ≈ a village.** **Urban cluster ≈ a city block / census tract.** Either way, geographically compact (a few hundred meters across).
- The geographic compactness is what makes "center one satellite tile on the cluster's coordinate" a defensible move. If clusters were spread across 50 km, the tile couldn't represent them.

### Why do we need both DHS GPS *and* Earth Engine?

**The question:** Aren't they both giving us spatial data?

**Why it matters:** Common confusion. Worth a clear data-flow picture.

**Answer:**
- They do **completely complementary** jobs:
  - **DHS GPS** tells us *where* each village is + urban/rural label + the cluster-ID join key. **It doesn't have imagery.**
  - **Earth Engine** tells us *what each village looks like from space*. **It doesn't know which patches of Africa contain DHS villages.**
- The **DHS GPS coordinate is the bridge** between the two. Without it, Earth Engine doesn't know which patch of Africa we care about for "village 47 in Kenya."
- The full data flow:
  1. DHS survey → wealth per cluster (the `y`)
  2. DHS GPS → lat/lon per cluster (the location key) + urban/rural flag
  3. Earth Engine, given that lat/lon → 8-channel satellite tile (the `x`)
  4. CNN learns `tile → wealth` on the resulting (image, number) pairs

### What is GeoPandas and why do we need it?

**The question:** Why not just use pandas?

**Why it matters:** Justifies bringing in another library.

**Answer:**
- **GeoPandas = pandas + spatial geometries.** A `GeoDataFrame` is just a regular pandas DataFrame with one extra column type — `geometry` — that holds spatial objects (Points, Polygons, Lines).
- We need it for **two reasons**:
  1. **Reading shapefiles.** DHS ships GPS data as shapefiles (`.shp`). Pandas can't read those natively; geopandas can, via `gpd.read_file(...)`.
  2. **Spatial operations later.** If we ever want "find clusters within 5km of this point" or spatial joins with administrative-region polygons, geopandas has the machinery.
- In this codebase, geopandas usage is currently minimal — just reading the DHS shapefile and pulling out `LATNUM`/`LONGNUM`/`URBAN_RURA` as ordinary columns. The geometry column sits there mostly unused for now.

### Why is the DHS GPS deliberately jittered?

**The question:** Why are the coordinates not exact?

**Why it matters:** Affects both the imagery quality AND a key fairness-audit experiment.

**Answer:**
- DHS deliberately offsets cluster GPS by up to **2 km in urban areas, 10 km in rural areas** for privacy.
- This adds noise to the input-output mapping. Yeh 2020's own jitter ablation found that adding extra jitter cost ~0.07 r² globally.
- **Uneven across urban/rural** (2km vs 10km) — the rural GPS is much noisier than urban. This is one mechanism that could explain part of the urban-rural fairness gap, and it's the basis for the GPS-jitter diagnostic experiment in `design.md` §7.6.

---

## 4. ML hygiene

### Why is it OK to standardize the wealth index across the whole pool (including test data)?

**The question:** Doesn't the standard ML rule say "fit on train only, transform test"?

**Why it matters:** Sharp critical-thinking question that distinguishes label processing from feature processing.

**Answer:**
- The "fit on train only" rule applies to **preprocessing of model inputs** (`x`) — fitting scalers, PCA on features, target encoders, etc.
- **The wealth index is the *label* (`y`), not a model input.** It's defined externally, before any train/test split, by a non-learning process. The CNN never sees how `y` was constructed; it just gets `(x, y)` pairs.
- Standardizing the target across all data is the same kind of thing as choosing units (Celsius vs Fahrenheit) — it's a definition choice about how to measure `y`.
- **Honest acknowledgment:** there's still a *technical* weak leakage — the PCA loadings were computed using test-fold households too. We deliberately accept this because (a) cross-country comparability requires consistent target definition across folds, (b) PCA loadings are extremely stable, (c) the leakage is on target definition, not on model inputs — much weaker than the textbook case, (d) Yeh did it this way too.
- **Where the rule still applies:** when `dataset.py` is built, satellite image normalization for non-RGB channels must use stats computed on training folds only.

### Why 5-fold cross-country and not 7- or 8-fold?

**The question:** Why this particular CV scheme?

**Why it matters:** Often asked in CV-design discussions.

**Answer:**
- 23 countries / 8 folds = ~3 test countries per fold → high-variance per-fold r².
- 5 folds = ~5 test countries per fold → more stable per-fold estimate.
- We use **Yeh 2020's exact Supplementary Table S2 fold assignment** for apples-to-apples comparison with the original.
- LOCO (23-fold leave-one-country-out) is a stretch goal — more rigorous but ~25–46 extra GPU-h.

---

## 5. Code-quality decisions made during the walkthrough

### Why semantic column names (has_electricity) instead of raw DHS codes (hv206)?

**The question:** Why rename in the output dataframe?

**Why it matters:** Demonstrates iterative code-quality improvement.

**Answer:**
- The `WealthIndexResult.feature_names` list now reads as `has_electricity`, `has_radio`, `finished_floor`, etc. — directly readable.
- PCA loadings become self-documenting: `has_electricity: +0.39` is immediately interpretable; `hv206: +0.39` requires looking up the DHS code book.
- Sanity-checking by eye becomes trivial — you can immediately spot a wrong-sign or wrong-magnitude loading.
- Cost: a one-line `BINARY_ASSETS = {code: name}` mapping at the top of `dhs.py`.

### Why filter `hv216` values ≥ 25 to NaN before clipping?

**The question:** What was the bug, and why does the fix matter?

**Why it matters:** Concrete data-quality story; shows attention to detail.

**Answer:**
- DHS uses special codes for `hv216` (number of sleeping rooms): **96** ("rooms not separated"), **97/98** ("don't know"), **99** ("missing").
- Naive `clip(0, 10)` would silently map those to 10, **fabricating a "10-room household" out of a don't-know answer.**
- The fix: mark any value ≥ 25 (clearly a special code or data-entry typo) as NaN *before* clipping. Real values pass through unchanged; specials are correctly marked as missing.
- This is the kind of silent-corruption bug that's invisible until someone looks specifically for it. Worth flagging in the methods section as an example of careful data hygiene.

### Why is no test coverage on `dhs.py` a real risk?

**The question:** We have working code on Kenya — what's missing?

**Why it matters:** Honest about current state. Demonstrates engineering awareness.

**Answer:**
- `dhs.py` contains several silently-wrong-able pieces of logic:
  - The PCA sign-flip (does electricity actually load positive on a different country's data?)
  - The PR→HR deduplication (does `(hv001, hv002)` uniquely identify a household in every survey?)
  - The DHS code-set lookups (do the code sets work correctly on a country with different code distributions?)
  - The missing-value helpers (do they handle all of pandas' missing-value flavors uniformly?)
- The Kenya end-to-end run is the *only* validation right now. Kenya could have been the easy case.
- Planned mitigation: when `tests/` is built out, add targeted tests for each defensive choice — especially the sign-flip (which a recent refactor would have silently broken without a test catching it).

---

## 6. Satellite imagery — the input pipeline (`earth_engine.py`)

### One-sentence summary of the input side

**The deliverable line (use this verbatim in the talk):**
> For each village, we collapse ~40 Landsat scenes captured over a 3-year window into a single 8-channel "typical view" image by taking the per-pixel median independently per band — that one image becomes the CNN's input, and the village's DHS-derived wealth index becomes its target.

**Why it matters:** Captures the *what*, the *why* (3-year window for cloud robustness), and the *how* (per-band median) in one breath.

### Why 224×224 pixels and 6.72 km on a side?

**The question:** Where do these specific numbers come from?

**Why it matters:** Two independent constraints intersect here — neither alone explains the choice.

**Answer:**
- Each pixel is 30 m × 30 m (Landsat's native resolution; can't get finer).
- 224 × 30 m = 6720 m = 6.72 km on a side.
- **Constraint 1 — DHS GPS jitter** offsets rural clusters up to 10 km, urban up to 2 km. A 6.72 km tile comfortably contains the actual village even with jitter.
- **Constraint 2 — 224×224 is ImageNet's standard input size.** Every torchvision model is pretrained at this resolution, so the tile drops straight into ResNet-18 with no resizing.
- The three constants must stay consistent: `2 × PATCH_HALF_SIDE_M = PATCH_SCALE_M × PATCH_SIZE_PX` → `2 × 3360 = 30 × 224`.

### What does each of the 8 bands tell us?

**The question:** Why these 8 bands?

**Why it matters:** Shows understanding of multispectral imaging, not just "we used what Yeh used."

**Answer:**

| Band | Measures | Contribution to wealth prediction |
|---|---|---|
| RED / GREEN / BLUE | visible light | rooftops, roads, visible structure |
| NIR | near-infrared | vegetation (plants reflect NIR strongly) — "crops vs bare dirt" |
| SWIR1 | shortwave infrared 1 | soil moisture, mineral composition |
| SWIR2 | shortwave infrared 2 | building materials (concrete/metal/thatch differ in SWIR) |
| TEMP1 | thermal infrared | urban heat island — cities are measurably hotter |
| NL | nightlights | electrification — the single strongest direct wealth proxy |

The first 7 come from Landsat (one satellite, one timestamp per scene). The 8th comes from a different satellite (DMSP or VIIRS).

### Why median composites over single scenes?

**Answer:**
- **Cloud-robust without per-pixel masking** — median statistically rejects cloud-contaminated values.
- **Cancels temporal noise** — haze, season, sun angle vary scene to scene; median averages them out.
- **Matches the target's time-scale** — DHS wealth is itself a multi-year-aggregated, place-level quantity.
- **Field standard** (Yeh, Jean) — required for apples-to-apples replication.
- Honest cost: synthesizes a "typical view" that corresponds to no specific instant.

### What alternatives to median did we consider? (be ready for "why not X")

| Strategy | Why we didn't use it |
|---|---|
| Mean | Sensitive to cloud outliers — one bright cloudy pixel skews it |
| Single best scene | Throws away ~95% of data; the chosen scene may still have local cloud |
| Cloud-shadow-masked mean | More principled but adds real engineering complexity |
| Quality mosaic | Per-pixel best-of-stack; spectrally inconsistent across bands |
| Greenest pixel (max NDVI) | Designed for vegetation studies; wrong for built-environment |
| Per-season composites | Triples data volume without clear gain for wealth |
| Harmonic regression | For change-over-time analysis (out of scope here) |

### Why is the chunky low-res nightlights band the *most* important band?

**The question:** If NL is so low resolution, why does it matter?

**Why it matters:** Counter-intuitive — a strong audience moment.

**Answer:** Yeh 2020's ablation (pooled r²):

| Model | Pooled r² |
|---|---|
| CNN MS+NL (full 8-channel) | 0.67 |
| **CNN NL only** (just the chunky 1-channel NL) | **0.66** |
| CNN MS only (7 high-res bands, no NL) | 0.62 |
| KNN scalar NL (avg brightness, no CNN at all) | 0.66 |

**Removing nightlights costs more r² than removing all seven high-res spectral bands.** Even the average nightlight intensity alone, with no CNN, gets 0.66. Why: nightlights directly measure electrification (the strongest wealth proxy), the discrimination needed is coarse ("dark village vs lit town vs city"), and direct measurements beat inferred ones.

### Why is the NL band so much lower resolution than Landsat?

**Answer:**
- VIIRS (nightlights satellite) is ~450 m/pixel; Landsat is 30 m/pixel. A 6.72 km tile = ~15×15 native VIIRS pixels, resampled up to 224×224 (hence the chunky look). DMSP-OLS (pre-2012) is even coarser, ~1 km.
- VIIRS sensors are designed for **darkness sensitivity** — bigger pixels collect more photons, giving measurable brightness at night. Finer resolution would make night measurements too noisy.
- Fine resolution isn't needed: electrification patterns are bigger than 450 m, so aggregate brightness per area is what matters.

### Why three different Landsat sensors (5, 7, 8)?

**Answer:**
- L5 (1984–2013), L7 (1999–now, "SLC-off" striping defect since 2003), L8 (2013–now, modern workhorse). Our DHS window (2009–2017) spans all three.
- Strategy: pick the newest sensor with coverage per window — 2013+ → L8, 1999–2012 → L7, pre-1999 → L5 (unused).
- **Band-naming nightmare:** each sensor numbers bands differently for the same wavelength (L8 RED = band 4; L5/L7 RED = band 3). `LANDSAT_BANDS` + `.select(sensor_bands, CANONICAL_BAND_NAMES)` rename hides this from everything downstream.

### Why DMSP for early years and VIIRS for later — and the catch?

**Answer:**
- DMSP-OLS (military weather sats, 1992–2013) — global, coarse, saturates in bright cities. VIIRS (NOAA Suomi-NPP, 2012+) — finer, wider dynamic range, no saturation.
- Yeh used DMSP for 2009–2011, VIIRS for 2012+ (we follow).
- **The catch:** DMSP and VIIRS have **incompatible value scales** — not directly comparable. Within each composite window we stay on one sensor, but cross-window nightlight comparisons are not directly meaningful.

### Earth Engine's lazy computation model

**The question:** Why do the composite functions return instantly?

**Why it matters:** Without this, people think we're already downloading data when we describe a composite.

**Answer:**
- Every `ee.ImageCollection(...).filterBounds(...).median()` chain is **just a recipe** — no pixels move.
- EE has petabytes; eager execution would melt it. **Pixels are only computed when materialized** — `.getDownloadURL()`, `.toDrive()`, `.getInfo()`.
- Same trick as Spark, Dask, SQL planners, React render trees: separate *describing what you want* from *computing it*.
- Practical consequence: building 19,000 image recipes in a loop is free; materializing them is the expensive part.

### Why two export paths — async to Drive vs sync direct download?

| Aspect | `export_cluster_to_drive` (async) | `download_cluster_tile_direct` (sync) |
|---|---|---|
| Use case | Bulk 19,000-cluster extraction | One-off debug / sanity-check / demo |
| Returns | Task handle (immediately) | File path (after blocking download) |
| Output | Google Drive | Local disk |
| Time per call | Milliseconds (just submits) | Minutes (waits for EE compute + download) |
| Failure mode | Task fails silently; poll status | Raises exception immediately |

The Day 5 Kenya test used the sync path (2 tiles, wanted locally, immediately). Bulk extraction will use the async path (throughput over latency).

### Why force keyword-only arguments for lat/lon?

**Answer:**
- The `*` first parameter (`def f(*, lat, lon, ...):`) forces callers to write `f(lat=..., lon=...)` — no positional calls allowed.
- **The bug it prevents:** (lat, lon) vs (lon, lat) confusion. EE itself uses (lon, lat) while humans say "lat, lon" — a recurring geospatial bug where imagery comes back from the wrong hemisphere. Forcing keyword names makes the order-confusion impossible.

### Why does the per-band median produce a "synthesized" image?

**Answer:**
- For one pixel location, median RED might come from a March 2013 scene while median NIR comes from a Sept 2014 scene — medians are computed independently per band.
- **Pixel locations are fixed** (same patch of ground); **per-band values at a location** can come from different scenes.
- Fine for most pixels (3-year ground stability); a known limitation in rapidly-changing areas (new construction, deforestation) where the composite represents neither before nor after cleanly. We accept it.

### "8 bands" — storage vs display vs modeling

**Answer:**
- **On disk:** one GeoTIFF with 8 spatially-aligned bands.
- **In our viz:** 8 separate grayscale panels (one per band), each percentile-stretched — which is why they look grey. Color only appears when 3 bands are combined into an RGB composite.
- **In the CNN:** one input tensor of shape `(8, 224, 224)` — all 8 channels seen simultaneously by the first conv layer.
- Humans can only directly see the 3 visible bands combined; the other 5 are outside human vision.

### The two seams: where DHS and Earth Engine connect

**The question:** Where in the code do DHS GPS and satellite imagery actually meet?

**Why it matters:** Clarifies the pipeline architecture; also flags that this glue isn't written yet.

**Answer:** Two connection points at different times:
- **Seam 1 — extraction time** (`scripts/03_download_imagery.py`, not yet written): loop over the DHS GPS DataFrame, call `export_cluster_to_drive(lat=row.LATNUM, lon=row.LONGNUM, year=..., cluster_id=...)`. The cluster_id is baked into the output filename.
- **Seam 2 — training time** (`data/dataset.py`, not yet written): the PyTorch Dataset joins the `.tif` files (by cluster_id in filename) to the wealth DataFrame (by cluster_id column), returning `(image, wealth)` pairs.

### Does excluding wealthy countries (South Africa, etc.) bias the PCA toward poorer countries?

**The question:** The pooled PCA is fit on the 23 sub-Saharan core countries; the wealthy OOD countries (South Africa, Namibia, Gabon...) are held out. Doesn't that bias the wealth axis and make the index "bad" on those countries?

**Why it matters:** Prime advisor question; goes to the heart of the OOD-test design.

**Answer — separate two claims:**
- *"The PCA axis is biased toward the poor"* — mostly **not** true. PC1 captures a single modern-infrastructure/SES factor (electricity–TV–fridge–finished-housing co-occur), and that covariance structure is stable across SSA, so adding wealthy countries shifts the mean / adds top-end variance but barely rotates the axis. The feature space also has a **ceiling**: built from ~15 *binary* assets, so the richest possible household is "all assets = 1" — wealthy countries don't open a new region, the 23 already contain maxed-out urban-elite households (Nairobi/Lagos/Accra). Ruler analogy: a 2 m ruler; the rich peg the top.
- *"The index/model performs worse on wealthy OOD countries"* — **true, and intended.** When an asset saturates (electricity ≈ 100% in South Africa) it stops discriminating wealth *within* that country, yet the frozen axis still weights it (0.374, learned from SSA). So the index compresses rich households together at the top (ceiling/compression). **Inherent to asset indices in wealthy populations — happens whether or not those countries are in the fit.** DHS's own index is famously weak at separating the rich from the very rich.

**Why we freeze the axis anyway (and must):**
- **Comparability:** one recipe = one ruler; "+1" must mean the same thing in every country. Refitting per experiment swaps rulers.
- **Faithfulness / no leakage:** textbook "fit preprocessing on train, transform test." OOD countries are *test* — project them onto the train-fit axis (reuse the saved 23-country mean/std + PC1 loadings), never refit.
- **It's the point of the OOD test:** "r² drops on wealthy OOD countries, partly from the asset ceiling" is a *finding* about generalization limits + the index's fairness across the wealth spectrum (it measures the poor more precisely than the rich) — not a bug. The core 23-country result is unaffected; all 23 are in the fit, mutually comparable.

**Run numbers (2026-05-21):** 355,445 households pooled → 13,634 clusters; PC1 = 28.7% variance; every asset loads positive (sign convention held); `has_bicycle ≈ 0` (wealth-neutral in SSA); urban/rural gap = 1.158σ.

---

## To be added (these come up in later modules)

As we walk through `dataset.py`, the model code, training, and the fairness audit, we'll keep adding. Open slots:

- **Dataset:** Why center-crop to exactly 224×224 (and why does EE return 225/226 sometimes)? Why the augmentation choices? How does the filename↔wealth join work in practice?
- **Splits:** Why Yeh's exact Supp Table S2 fold assignment? Why country-level (not village-level) folds for the cross-country protocol?
- **Model:** Why ResNet-18? Why the scaled-ImageNet 8-channel init trick? Why a scalar regression head?
- **Training:** Why MSE? Why Adam? Why the 0.96/epoch lr decay? Why batch size 64?
- **Fairness audit:** Why bootstrap CIs? Why permutation tests? Why these subgroups? Why the targeting-accuracy stratification?
- **Uncertainty extension:** Why MC-dropout vs ensembles or full Bayesian? Why 50 inference passes?
- **Temporal extension:** Why the P1+P2 → P3 split? Why decompose drift into three components?

---

*Last updated: pooled wealth index computed across all 23 countries (355,445 households → 13,634 clusters, PC1 = 28.7% variance, urban/rural gap 1.158σ); added the wealthy-OOD-countries PCA-bias Q&A. Will keep growing as we move through the codebase.*
