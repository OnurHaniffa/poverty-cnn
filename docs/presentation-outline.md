# Lab Presentation — Blueprint (~25-30 min, ~17 slides)

**Audience:** mixed lab group, ML-literate enough to know what a CNN is (don't teach
it) but everything else taught from intuition. **Goal:** they understand *as you talk.*
**Rhythm for every concept:** Why → What → Show (one analogy OR one picture) → So what.
**Running example:** a single Kenyan cluster (Nairobi = urban, Turkana = rural).

---

## ACT 1 — Why are we here (2 slides)

**S1 · The problem.** Poverty data is scarce and expensive — household surveys cost
millions and happen every ~5 years, so most of the map is blank or stale. But aid
targeting needs to know *where the poor are*. **Question:** can a *free* satellite
image, available everywhere, substitute for a survey?
- *Visual:* a sparse survey-coverage map next to a satellite image.

**S2 · The idea, in one picture.** `satellite image → neural network → wealth number`.
That's the whole project. The talk answers two things: *where do the "right answers"
come from?* and *does it work — and for whom?*
- *Visual:* `diagrams/01_big_picture.png`.

## ACT 2 — The ingredients (4 slides)

**S3 · The answer key: DHS surveys.** Gold-standard household surveys (USAID). Unit =
a **cluster** ≈ a village (~25 households). Each has a **GPS point** — but DHS *fuzzes*
it (up to 2 km urban / 5 km rural) for privacy. **So-what:** that's exactly why our
satellite tiles are a wide **6.7 km box** — big enough to still contain the real village.
- *Analogy:* the survey is the "answer key" we train and grade against.

**S4 · The hard part: there is no "wealth" column.** DHS gives asset *checkboxes* —
electricity? TV? finished floor? a car? — not a wealth number. **Motivates** the next slide.
- *Visual:* `eda/03_asset_prevalence.png` (or a sample household row of 0/1s).

**S5 · PCA: finding the hidden "wealth axis."** PCA finds the single combination of the
15 assets that best explains who has *more stuff*. That combination (PC1) **is** wealth —
we know because every asset loads *positive* (a "more-of-everything" axis). Average it to
the village = our target `y`.
- *Analogy:* one **ruler** that puts every household on the same scale; or a
  **credit-score formula** discovered from the data, not hand-set.
- *Visual:* `eda/02_pca_loadings.png`. *Pre-empt the FAQ:* "only 28%?" → it's 4× the next
  factor; validity comes from the coherent loadings + the urban/rural gap, not the %.

**S6 · The input: 8 channels (the money-shot).** Show a **real Nairobi (urban) and
Turkana (rural)** tile — all 8 bands side by side, then collapsed to the human-viewable
RGB. The CNN reads *all 8 layers together* and outputs *one number*.
- *Analogy:* 8 "senses" — visible color, near/short-wave infrared (vegetation, rooftops),
  thermal (heat), and **night-lights** (electrification).
- *Visual:* `tile_viz/nairobi_all_bands.png` + `turkana_all_bands.png` (+ composites).
  *(Secretly plants the urban/rural theme for the audit later.)*

## ACT 3 — How we do it & test it honestly (3 slides)

**S7 · The model (kept short).** A **ResNet-18** (standard CNN), two tweaks: input rebuilt
for 8 channels (not 3), output is one number (not categories). Trained from scratch. *Move on.*

**S8 · The honest test: cross-country.** If we split villages randomly, the model can
*memorize* a country and cheat. So we **train on ~18 countries and test on 5 it has never
seen** — rotated so every country is tested once. **So-what:** every number we report is on
*unseen countries* — the real-world deployment question.
- *Analogy:* an exam with questions you never studied — the only fair test.

**S9 · ★ How we measure success — the metrics ★** *(built around ONE example: 5 villages,
true vs predicted)*:
- **r²** = *what fraction of the spread did we capture?* (0 = guessing the average, 1 =
  perfect). **But** it's misleading alone — it mixes "how wrong" with "how spread out."
- **MAE** = *on average, how many points are we off?* — honest, in wealth units.
- **Spearman** = *did we get the village* ***order*** *right?* — pure ranking.
- **Punchline:** they answer *different* questions, so we report all three. **For finding
  the poorest, ranking (Spearman) is what matters** — you want the order, not the exact score.
- *Analogy:* grading students — MAE = average points off; Spearman = did you *rank* them
  right for the scholarship; r² = how much of the class's variation you explained.

## ACT 4 — What we found (5 slides)

**S10 · Does it work? (Replication.)** Pooled r² **0.57**, mean **0.52** on unseen
countries — vs the landmark paper's 0.67/0.70. We *beat the dumb floor* (0.25), so it's
genuinely reading wealth from space. The gap is **data, not method**: we used *one* survey
round/country; they pooled *many*. (One honest line; don't over-dwell.)

**S11 · What carries the signal? Nightlights dominate.** Night-lights *alone* ≈ the full
8-channel model. Lights = electrification = the strongest village-wealth signal. Daytime
imagery adds little here (it's data-hungry). *Sets up the limitation to come.*

**S12 · The audit, part 1: urban vs rural.** *Callback to Nairobi/Turkana.* Equal *absolute*
accuracy (MAE), but the model **ranks rural villages worse** (Spearman 0.41 vs 0.54) — and
this holds in 21/23 countries. It's better in cities.
- *Visual:* `fairness/01_urban_rural.png`.

**S13 · The audit, part 2: the poorest.** The model **regresses to the mean** — it predicts
the poorest villages as *richer than they are*, and its error is *largest exactly there*.
"Confidently wrong about the poorest." Consequence: targeting the poorest 20%, it catches
only ~23% of them.
- *Visual:* `fairness/05_calibration.png` + `fairness/06_bias_by_wealth.png`.

**S14 · ★ The punchline: a fundamental limit ★.** Three independent findings, one cause:
- *Confidently wrong* (the audit) →
- *Undetectable* (no uncertainty method — ensembles, MC-dropout, learned-variance — flags it) →
- *Unfixable* (reweighting the loss toward the poor doesn't repair it; it just destabilizes).
- **Because the signal isn't there:** night-lights are uniformly dark below the poverty line,
  so the satellite *cannot separate the extreme poor*. No amount of modeling fixes missing
  information. **THE takeaway slide.**

## ACT 5 — Close (3 slides)

**S15 · What's next: the "full" study.** This was a deliberate single-round *pilot*. Running
now: a 3× multi-round dataset (to close the data gap), plus temporal drift (does the gap widen
over time?) and an out-of-distribution test on wealthier countries.

**S16 · Contributions & honest positioning.** Faithful modern replication + the most
comprehensive **fairness + uncertainty audit** on this 23-country data (extends prior work
from 10 → 23 countries). Most novel: the *equity-framed uncertainty result*. Honest: this is
an **audit**, not a new architecture — and we don't claim a SOTA number.

**S17 · The one-liner + questions.** *"Satellite poverty maps are good for ranking regions —
not for last-mile targeting of the extreme poor — and standard uncertainty tools can't warn
you when they're wrong about the neediest."* Thank you / questions.

---

### Assets ready: diagrams (3), tile_viz (Nairobi+Turkana ×8 bands), eda (10),
### fairness (6), uncertainty (6), mitigation (1). Need to make: the metrics worked-example
### graphic (S9), and possibly the survey-coverage map (S1).
