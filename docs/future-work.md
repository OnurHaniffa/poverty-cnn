# Future Work & Improvement Backlog

Improvements identified but **deliberately deferred** — revisit once the core replication + fairness audit are working. Capturing them here so they're not lost.

## 1. Imbalance handling (the big one — ties into the fairness story)

We currently handle **no** imbalance (matches Yeh's plain-MSE baseline). Three real ones exist:

1. **Skewed wealth distribution** — right-skewed (many poor villages, few rich). MSE can over-attend to the rich tail.
2. **Country sample-size imbalance** — Kenya 1,585 clusters vs. Mali 328. Pooled MSE optimizes harder for big countries → **likely a direct cause of the per-country r² disparity** we observed (e.g., Benin −0.40, Togo +0.78).
3. **Urban/rural imbalance** — ~8,600 rural vs. ~5,000 urban clusters.

**Proposed handling (when we get to it):**
- Loss reweighting inversely by country/group size, **or** class-balanced batch sampling (equal countries per batch).
- **Strategic framing — do NOT silently "fix" #2/#3.** They are likely *root causes* of the fairness gaps we are auditing. Plan:
  1. Audit the **unweighted** model first — the disparity *is* the finding (contribution #2).
  2. Then test reweighting / balanced sampling as a **mitigation intervention** — does balancing shrink the per-country gap without tanking overall r²?
  - That "diagnose-then-mitigate" arc is a genuinely novel, publishable angle for contributions #2/#3.

## 2. Eval / metric upgrades (cheap — fold into the eval module)

- Log **train** r²/loss alongside val each epoch → *visualize the overfitting gap* (currently only val is logged; early stopping protects us but we can't yet draw the gap).
- Report **RMSE + MAE** (interpretable, in wealth-index units) and **Spearman rank correlation** (ranking quality, relevant to targeting) — not just r²/Pearson.
- **Targeting-accuracy** framing: precision/recall of correctly flagging the poorest X% of villages (the aid-allocation simulation, contribution #3) — this is where classification-style metrics legitimately return.

## 3. Training / rigor upgrades (tuning stage)

- Put **SGD+momentum** in the Optuna search space — it often *generalizes* better than Adam for image CNNs (Adam was a pragmatic baseline choice).
- Consider **nested cross-validation** for hyperparameter search (inner CV inside each outer training fold) — professor prefers it; ~5× compute, so a deliberate trade-off vs. our current train/val/test-per-fold.
- **Pretrained-adapted ResNet** as an ablation — measure whether ImageNet init beats from-scratch on our 8-channel data.

---
*Logged 2026-05-25, mid-baseline. None of these block the core deliverables; they are "add the jazz" after the spine is done.*
