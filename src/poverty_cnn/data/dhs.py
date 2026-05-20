"""Load DHS household survey data and compute the pooled asset wealth index.

Follows Yeh et al. (2020) §Methods: pool households across all study
countries, run PCA on standardized asset features, take the first principal
component as the household wealth index, then average to cluster level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

# Binary asset-ownership indicators (1 = owns, 0 = doesn't).
# DHS Recode 6 variable codes (2009+). Older rounds may have variants.
# Mapping is `DHS code → semantic feature name`. Using semantic names downstream
# keeps the PCA loadings readable (e.g. "has_electricity: 0.39" instead of
# "hv206: 0.39") in WealthIndexResult.feature_names.
BINARY_ASSETS = {
    "hv206":  "has_electricity",
    "hv207":  "has_radio",
    "hv208":  "has_tv",
    "hv209":  "has_fridge",
    "hv210":  "has_bicycle",
    "hv211":  "has_motorcycle",
    "hv212":  "has_car",
    "hv221":  "has_landline",
    "hv243a": "has_mobile",
}

# Numeric / count variables
COUNT_ASSETS = {
    "hv216": "sleeping_rooms",   # number of rooms used for sleeping
}

# Categorical variables we recode into a single binary "is improved?" feature.
# This sidesteps country-specific ordinal scoring at the cost of some signal.
# See `improved_water_source`, `improved_toilet`, etc. below.
CATEGORICAL_VARS = ["hv201", "hv205", "hv213", "hv214", "hv215"]

# DHS standard "improved water source" codes
# (piped, tubewell, protected well, protected spring, rainwater, bottled, sachet)
IMPROVED_WATER_CODES = {11, 12, 13, 14, 21, 31, 41, 51, 61, 71, 72}

# DHS standard "improved sanitation" codes
# (flush toilets, VIP latrines, pit with slab, composting)
IMPROVED_TOILET_CODES = {11, 12, 13, 14, 15, 21, 22, 41}

# "Finished" floor / wall / roof materials
# These vary slightly by survey but the broad codes are stable.
FINISHED_FLOOR_CODES = {30, 31, 32, 33, 34, 35, 36, 37}    # 30-series
FINISHED_WALL_CODES = {30, 31, 32, 33, 34, 35, 36, 37}     # 30-series
FINISHED_ROOF_CODES = {30, 31, 32, 33, 34, 35, 36, 37}     # 30-series


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_dhs_hr(filepath: str | Path) -> pd.DataFrame:
    """Load a single DHS Household Recode (HR) file in Stata .DTA format.

    Columns are lowercased on load to normalize across surveys.
    Categorical values are kept as integer codes (not labels) so we can
    apply our own standardized recoding across countries.
    """
    filepath = Path(filepath)
    df = pd.read_stata(filepath, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    return df


def load_dhs_pr_as_hr(filepath: str | Path) -> pd.DataFrame:
    """Load a DHS Person Recode (PR) file and reduce to one row per household.

    PR has one row per individual; HR has one row per household. Asset
    variables in PR are household-level (same value across all members of
    a given household), so deduplicating by (hv001, hv002) gives the same
    rows as the HR file.

    Useful when only PR is available locally (DHS sometimes bundles PR
    with the GPS download instead of HR).
    """
    filepath = Path(filepath)
    df = pd.read_stata(filepath, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    deduped = df.drop_duplicates(subset=["hv001", "hv002"], keep="first").copy()
    deduped = deduped.reset_index(drop=True)
    return deduped


def load_dhs_gps(filepath: str | Path):
    """Load a DHS Geographic (GE) shapefile of cluster GPS points.

    Returns a GeoDataFrame with one row per cluster, including:
    - DHSCLUST (cluster ID = hv001 in HR file)
    - URBAN_RURA ('U' or 'R')
    - LATNUM, LONGNUM (jittered GPS)
    - geometry (Point)
    """
    import geopandas as gpd
    gdf = gpd.read_file(filepath)
    return gdf


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _binary_safe(series: pd.Series) -> pd.Series:
    """Convert a DHS-coded variable to 0/1, treating missing/special as NaN.

    DHS codes: 0 = no, 1 = yes, 9 = missing, 99 = not applicable.
    """
    return series.map({0: 0.0, 1: 1.0}).astype(float)


def _is_in_set(series: pd.Series, code_set: set[int]) -> pd.Series:
    """Return 1.0 where series matches any code in `code_set`, 0.0 otherwise, NaN if missing."""
    return series.where(series.notna(), other=np.nan).map(
        lambda v: 1.0 if (pd.notna(v) and int(v) in code_set) else (0.0 if pd.notna(v) else np.nan)
    )


def extract_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the wealth-index feature matrix from a DHS HR DataFrame.

    Returns one row per household with semantically-named asset features.
    Missing values stay as NaN — caller imputes or drops.
    """
    out = pd.DataFrame(index=df.index)

    # 1. Binary asset ownership — emit semantically-named columns.
    for code, name in BINARY_ASSETS.items():
        if code in df.columns:
            out[name] = _binary_safe(df[code])
        else:
            out[name] = np.nan   # variable not collected in this round

    # 2. Number of sleeping rooms.
    # DHS uses special codes (96 = "rooms not separated", 97/98 = "don't know",
    # 99 = "missing") in addition to genuine counts. A naive `clip(0, 10)` would
    # silently map those to 10, fabricating a "10-room household" out of a
    # don't-know answer. We mask anything >= 25 (clearly a special code or a
    # data-entry typo) to NaN first, then clip the plausible range.
    if "hv216" in df.columns:
        rooms = df["hv216"].where(df["hv216"] < 25, other=np.nan)
        out["sleeping_rooms"] = rooms.clip(0, 10).astype(float)

    # 3. Recoded categoricals → binary "improved" / "finished" indicators
    if "hv201" in df.columns:
        out["improved_water"] = _is_in_set(df["hv201"], IMPROVED_WATER_CODES)
    if "hv205" in df.columns:
        out["improved_toilet"] = _is_in_set(df["hv205"], IMPROVED_TOILET_CODES)
    if "hv213" in df.columns:
        out["finished_floor"] = _is_in_set(df["hv213"], FINISHED_FLOOR_CODES)
    if "hv214" in df.columns:
        out["finished_wall"] = _is_in_set(df["hv214"], FINISHED_WALL_CODES)
    if "hv215" in df.columns:
        out["finished_roof"] = _is_in_set(df["hv215"], FINISHED_ROOF_CODES)

    return out


# ---------------------------------------------------------------------------
# Pooled wealth index (PCA across all 23 countries)
# ---------------------------------------------------------------------------

@dataclass
class WealthIndexResult:
    """Output of the pooled PCA wealth-index computation."""

    household_index: pd.Series          # per-household, indexed by (country, hh_id)
    cluster_index: pd.DataFrame         # per-cluster aggregation
    pca_components: np.ndarray          # PC1 loadings on the standardized features
    feature_names: list[str]            # order of features in pca_components
    explained_variance_ratio: float     # variance explained by PC1


def pooled_wealth_index(
    country_features: dict[str, pd.DataFrame],
    country_clusters: dict[str, pd.Series],
) -> WealthIndexResult:
    """Compute the pooled asset wealth index across all study countries.

    Args:
        country_features: {country_code: feature_df} where each feature_df
            is the output of `extract_asset_features` for that country.
            Index is the household identifier; rows must align with the
            corresponding `country_clusters[country_code]` series.
        country_clusters: {country_code: cluster_id_series} giving the
            cluster (HV001) for each household. Same length and order as
            the matching `country_features[country_code]`.

    Returns:
        WealthIndexResult with the per-household wealth index, per-cluster
        averages, and PCA diagnostics.
    """
    from sklearn.decomposition import PCA

    # 1. Pool households across countries, tagging origin
    frames = []
    cluster_frames = []
    for country, feat in country_features.items():
        tagged = feat.copy()
        tagged["country"] = country
        clusters = country_clusters[country].rename("cluster_id")
        tagged["cluster_id"] = clusters.values
        frames.append(tagged)
        cluster_frames.append(pd.DataFrame({"country": country, "cluster_id": clusters.values}))
    pooled = pd.concat(frames, ignore_index=True)

    # 2. Drop households missing too many features
    feature_cols = [c for c in pooled.columns if c not in {"country", "cluster_id"}]
    coverage = pooled[feature_cols].notna().mean(axis=1)
    keep = coverage >= 0.7   # require at least 70% of features present
    pooled = pooled.loc[keep].copy()

    # 3. Impute remaining NaNs with column mean (within country, then global fallback)
    for country in pooled["country"].unique():
        mask = pooled["country"] == country
        country_means = pooled.loc[mask, feature_cols].mean()
        pooled.loc[mask, feature_cols] = pooled.loc[mask, feature_cols].fillna(country_means)
    # Final fallback for features missing in entire country
    pooled[feature_cols] = pooled[feature_cols].fillna(pooled[feature_cols].mean())

    # 4. Standardize features (z-score across pooled sample)
    X = pooled[feature_cols].values.astype(float)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    # 5. PCA, take PC1
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_std).ravel()

    # 6. Standardize PC1 to mean 0, std 1 for clean interpretation
    pc1 = (pc1 - pc1.mean()) / pc1.std(ddof=0)

    # 7. Sign convention: positive PC1 = wealthier.
    # Force "has_electricity" to load positively as a sanity convention.
    # (Pinned to the semantic name set by extract_asset_features.)
    if "has_electricity" in feature_cols:
        elec_idx = feature_cols.index("has_electricity")
        if pca.components_[0, elec_idx] < 0:
            pc1 = -pc1
            pca.components_ = -pca.components_

    pooled["wealth_index"] = pc1

    # 8. Aggregate to cluster level
    cluster_index = (
        pooled.groupby(["country", "cluster_id"], as_index=False)["wealth_index"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "wealth_index_mean", "count": "n_households"})
    )

    return WealthIndexResult(
        household_index=pooled["wealth_index"],
        cluster_index=cluster_index,
        pca_components=pca.components_[0],
        feature_names=feature_cols,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
    )
