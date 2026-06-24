# EDA Summary — pooled wealth index (23 countries)

- Households pooled: **355,445**  |  clusters: **13,634**  |  countries: 23
- Coverage filter (>=70% features present) dropped only ~30 households.
- Missing values in the final cluster table: **0**.
- Households per cluster: mean 26.1, median 26 (DHS design ~25-30).

## PCA / variance explained
- **PC1 = 28.7%** of total variance.
- Random/independent baseline (1/15) = **6.7%**  ->  PC1 is **4.3x** the baseline.
- PC2 = 9.6%, PC3 = 8.1% (sharp drop-off after PC1 = one dominant factor).
- All assets load positive except `has_bicycle` (~0, wealth-neutral) -> a single coherent wealth dimension.

### Why 28% is plenty for a wealth index
1. Binary survey items carry huge item-specific noise; first-factor variance of 20-40% is the norm and is *strong*, not weak. The '50%+' rule comes from continuous, tightly-correlated data.
2. The right benchmark is the random baseline (6.7%), not 50%. PC1 captures ~4x that.
3. Variance-explained measures *compactness*, not *validity*. PC1 is validated by: coherent positive loadings, a sensible country ranking, and a large correct-direction urban/rural gap.
4. The other ~71% is mostly NON-wealth (country-specific quirks, item noise) we deliberately exclude.
5. It is the field-standard DHS Wealth Index method (Filmer-Pritchett 2001) and exactly what Yeh 2020 used.

## Wealth distribution
- Cluster wealth: mean 0.011, std 0.841, skew 0.34.
- Cluster-level std < 1 because we standardize at household level then average; ~71% of wealth variance is between-village, ~29% within-village.
- Urban mean +0.742 vs rural -0.416 = **1.158 sigma gap**.

## Figures (results/figures/eda/)
01 scree · 02 loadings · 03 asset prevalence · 04 household wealth · 05 cluster wealth · 06 by-country · 07 urban/rural · 08 clusters-per-country · 09 households-per-cluster · 10 cluster map