# Speaker Notes — Poverty-from-Space Defense (33 slides)

**Throughline (hold this in your head the whole time):**
*"I faithfully replicate satellite poverty-mapping, then audit who it fails — and find it systematically fails the poorest, because the signal it leans on is physically absent where they live."*

Pacing: ~12–15 min talk. Spend most time in Act 3 (it works) and Act 4 (the turn). Dividers are 5-second breaths.

---

## ACT 1 — THE PROMISE

**1 · Title** — *Set the stage + credibility.*
"My project measures poverty from space — predicting how wealthy a village is from satellite imagery, across 23 African countries. I replicate a landmark 2020 result, then audit it: does it work *fairly*, especially for the poorest?" (Name advisor + internship.)

**2 · What this is / isn't** — *Scope + humility; pre-empt "did you beat SOTA?"*
"Two things up front. This IS a faithful replication plus a fairness/uncertainty/generalization audit. It is NOT a new architecture or a state-of-the-art claim — I use a plain ResNet-18. The science is in the data and the honest evaluation, not the model."

**3 · Divider 01 — The Promise** — "Let me start with why anyone thinks this is even possible."

**4 · The problem** — *Why it matters: the poverty data gap.*
"To fight poverty you must know where it is. But household surveys are expensive and rare — many countries go 10+ years between them. So for most of the map, most of the time, we're flying blind. That's the gap satellites could fill: they image everywhere, cheaply, repeatedly."

**5 · The idea** — *Wealth leaves visible fingerprints.*
"Wealth leaves physical fingerprints visible from orbit — metal roofs, paved roads, farmland, and especially electric light at night. So: feed satellite imagery to a neural network, have it predict wealth. That's what Yeh et al. showed in 2020 — and what I rebuild."

---

## ACT 2 — THE INGREDIENTS

**6 · Divider 02 — The Ingredients** — "Three ingredients: ground-truth labels, the right images, and a model."

**7 · DHS (ground truth)** — *Where the labels come from.*
"Labels come from the DHS surveys — the gold standard. They visit household clusters and record what people own. I use 23 countries, ~36,000 villages. This is the 'truth' the model learns from and is graded against."

**8 · Data cleaning** — *The unglamorous quality work.*
"Survey data is messy. I removed bad-data records — 'don't-know' codes, households missing too many fields, broken GPS, cloud-covered tiles. Key point: I removed *errors*, not extreme-but-real villages — keeping the genuine poorest and richest is essential, because they're what the audit is about."

**9 · PCA wealth index** — *There's no 'wealth' column — we construct it.*
"There's no wealth number in the data, just a list of assets. So, like a credit score, I use PCA to fuse ~15 assets into one index. The first principal component IS the wealth axis — TV, electricity, finished floors all load positively; 14 of 15 assets point the same way. Built with the standard DHS method, pooled across countries for a common scale." *(If pushed: "and I confirmed the findings survive a within-country re-index — backup slide.")*

**10 · EDA box plots** — *Know the data before modeling.*
"Wealth varies hugely both between and within countries, and it's right-skewed. The 'outlier' dots are real villages at the tails — not errors. That matters: those tails — the poorest — are exactly where the model will struggle."

**11 · 8 channels** — *The input is one 8-channel image.*
"The input isn't an ordinary photo — it's one image, eight channels deep, like RGB but with eight measurements per pixel: colour, near-infrared, shortwave infrared, thermal, and night-lights. Each exposes a different physical signature of wealth — same village, seen eight ways. Each channel is a cloud-free 3-year median."

**12 · The model** — *A deliberately ordinary CNN.*
"An off-the-shelf ResNet-18, trained from scratch. I change only the input (8 channels) and the output (one wealth number). MSE loss, Adam, early-stopping on validation — standard choices, because the point is the data and the evaluation, not the network."

---

## ACT 3 — DOES IT WORK?

**13 · Divider 03 — Does it work?** — "Does it work — and I'm very careful about HOW I measure."

**14 · CV ladder** — *How you split decides whether you're fooling yourself.*
"Shuffle villages randomly → r² 0.73. But that's inflated — villages from the same country leak across train and test. The honest test is leave-country-out: hold out WHOLE countries. That drops it to 0.57. Every number in this talk is the honest, country-held-out one."

**15 · Three metrics** — *Ranking is what matters; a model can rank well while compressing values.*
"r² = how much spread we capture; MAE = how many points off; Spearman = did we get the ORDER right. For targeting you need the order — so Spearman is my headline. And watch the example: the model squeezes values toward the middle, so r² suffers but ranking stays perfect. That squeeze is the first hint of a bias I return to."

**16 · Overfit + data-scaling** — *Not overfitting; the ceiling is data.*
"Two honesty checks. Left: training hits 0.81 but validation plateaus at 0.55 — I early-stop and report only validation, so it's not overfitting. Right: the curve is still rising with more data — so I'm below Yeh because of data budget, not a worse model. But — more data lifts the *average*, not the poorest. Hold that."

**17 · Benchmark results** — *It genuinely works, at benchmark level.*
"The honest, country-held-out numbers, with confidence intervals: Pearson r 0.76, r² 0.57, Spearman 0.72 — in the range of published models, on a harder split. So: yes, it works. That's the first half of the story."

**18 · Night-lights dominate** — *One channel does most of the work.*
"Which channel carries it? I retrained dropping each. Remove colour, infrared, thermal — ranking barely flinches. Remove NIGHT-LIGHTS and it collapses, because lights are a direct proxy for electricity. But that dependence is a setup: what happens where there are no lights?"

**19 · OOD generalization** — *Ranking transfers to unseen countries.*
"I froze the model and tested 6 brand-new countries. Ranking transfers — Spearman 0.48 to 0.81; Gabon even beats the in-distribution average. The learned notion of wealth genuinely carries over. The 'it works' case is strong. Now the turn."

---

## ACT 4 — THE TURN

**20 · Divider 04 — The Turn** — "It works — on average. But averages hide who it fails. Who does it work FOR?"

**21 · Cities > villages** — *It ranks the rural poor worst.*
"Split urban vs rural: same absolute error, but it can't RANK rural villages — Spearman 0.54 urban vs 0.38 rural. They're bunched near the bottom and have few lights to read. And rural is the majority — 8,473 villages — exactly who targeting must reach."

**22 · Confidently wrong about the poorest** — *Systematic regression to the mean.*
"Random error, or systematic? Systematic. A clean staircase: it over-predicts the poorest by +0.62, under-predicts the richest. Slope 0.60 — it drags every guess toward the average. So the poorest are predicted RICHER than they are — lifted out of the danger zone on paper. That's dangerous for finding them."

**23 · Calibration ≠ equity** — *It doesn't even know it's wrong about the poor.*
"Could uncertainty save us — flag low confidence on the poor? I tested three methods. None flag the poorest; they're overconfident exactly where they're most wrong. Why? Error = noise + bias + variance. Uncertainty sees variance. The poorest's error is BIAS — structurally invisible. Calibration is not equity. *(This is my most novel result.)*"

**24 · Nothing fixes it** — *Standard de-biasing fails — not a tuning bug.*
"Fixable artifact? I ran the field's standard cures — reweighting, Balanced-MSE, tripling the data. Reweighting barely moved it; Balanced-MSE made it worse; more data didn't touch the poorest. The *level* is trivially correctable, but no loss recovers RANKING among the poorest. A signal limit, not a tuning problem."

**25 · The lights are dark** — *The physical root cause.*
"Why? Here's the cause. Night-light brightness vs wealth is a hockey stick: below the median there's almost NO light — the poorest half of villages are physically dark. The model's main signal doesn't exist where the poor live. You can't rank what you can't see. No loss fills a gap in the data."

---

## ACT 5 — SO WHAT?

**26 · Divider 05 — So what?** — "What does this mean for actually using these maps?"

**27 · Targeting recall** — *It misses about half the neediest.*
"Target the poorest 20% of villages for aid, and you'd correctly reach just 49% of them — 2.5× better than random, but still missing half the neediest. For ranking regions: useful. For finding the poorest to help: not ready."

**28 · What to remember** — *The one-sentence takeaway.*
"If you take one thing: these maps are good at RANKING regions but dangerous for TARGETING the poorest — and that's not a bug to fix, it's a limit of what's visible from space."

**29 · Contributions** — *What's new.*
"A faithful 23-country replication; plus an audit showing the failure is structural — across fairness, uncertainty, and mitigation, with a physical root cause. Three independent lines of evidence, two replications. Not a new model — a clearer picture of an existing one's limits."

**30 · Thank you / back-pocket** — *Close + invite questions.*
"Thank you — happy to take questions, and I've got the key numbers in my back pocket." (r 0.76, r² 0.57, Spearman 0.72, poorest bias +0.62.)

**31 · References** — leave up during Q&A.

---

## BACKUP (only if asked)

**32 · Controls & robustness** — night-lights-only baseline (ranks like the CNN, but linear r² 0.16); pretrained = −0.03 r² (gap is data, not architecture); spatial autocorrelation (Moran's I, whole-country holdout limits leakage); slope is level-correctable, not rank-correctable.

**33 · Within-country robustness** — *for the pooled-PCA question:* strip the between-country signal and all three findings survive — Spearman 0.70, regression slope 0.65 (23/23 countries), poorest-bias +0.41 (21/23). Not an artifact of the pooled index.

---

### The three questions you WILL get — have these ready
1. **"Isn't the pooled wealth index a problem?"** → Backup 33: findings survive within-country.
2. **"Isn't this just a night-lights regression?"** → Backup 32: NL ranks well but captures no *level*; CNN adds the non-linear level + multispectral.
3. **"Why below Yeh's 0.70?"** → Slide 16: data budget, not architecture (pretraining is −0.03).
