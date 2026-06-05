"""Yeh 2020 5-fold cross-country split (== WILDS PovertyMap _SURVEY_NAMES_2009_17).

Vendored verbatim so the partition is explicit and reviewable, with no runtime
`wilds` dependency. Source: wilds 2.0.0 wilds/datasets/poverty_dataset.py.
Each country is in `test` exactly once across the 5 folds (apples-to-apples with
Yeh 2020 Supplementary Table S2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Our 23 DHS country codes -> WILDS country-name strings.
COUNTRY_CODE_TO_NAME = {
    "AO": "angola", "BJ": "benin", "BF": "burkina_faso", "CM": "cameroon",
    "CI": "cote_d_ivoire", "CD": "democratic_republic_of_congo", "ET": "ethiopia",
    "GH": "ghana", "GN": "guinea", "KE": "kenya", "LS": "lesotho", "MW": "malawi",
    "ML": "mali", "MZ": "mozambique", "NG": "nigeria", "RW": "rwanda",
    "SN": "senegal", "SL": "sierra_leone", "TZ": "tanzania", "TG": "togo",
    "UG": "uganda", "ZM": "zambia", "ZW": "zimbabwe",
}
NAME_TO_CODE = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}

# test + val country names per fold (train = all - test - val). From WILDS.
_FOLDS_NAMES = {
    "A": {"test": ["angola", "cote_d_ivoire", "ethiopia", "mali", "rwanda"],
          "val":  ["benin", "burkina_faso", "guinea", "sierra_leone", "tanzania"]},
    "B": {"test": ["benin", "burkina_faso", "guinea", "sierra_leone", "tanzania"],
          "val":  ["cameroon", "ghana", "malawi", "zimbabwe"]},
    "C": {"test": ["cameroon", "ghana", "malawi", "zimbabwe"],
          "val":  ["democratic_republic_of_congo", "mozambique", "nigeria", "togo", "uganda"]},
    "D": {"test": ["democratic_republic_of_congo", "mozambique", "nigeria", "togo", "uganda"],
          "val":  ["kenya", "lesotho", "senegal", "zambia"]},
    "E": {"test": ["kenya", "lesotho", "senegal", "zambia"],
          "val":  ["angola", "cote_d_ivoire", "ethiopia", "mali", "rwanda"]},
}


def fold_ids() -> list[str]:
    return list("ABCDE")


def countries_for(fold: str, role: str) -> set[str]:
    """Return the set of country CODES for a fold/role."""
    f = _FOLDS_NAMES[fold]
    test = {NAME_TO_CODE[n] for n in f["test"]}
    val = {NAME_TO_CODE[n] for n in f["val"]}
    if role == "test":
        return test
    if role == "val":
        return val
    if role == "train":
        return set(COUNTRY_CODE_TO_NAME) - test - val
    raise ValueError(f"role must be train/val/test, got {role!r}")


def clusters_for(metadata: pd.DataFrame, fold: str, role: str) -> np.ndarray:
    """Integer row positions of `metadata` whose `country` belongs to fold/role."""
    codes = countries_for(fold, role)
    return np.flatnonzero(metadata["country"].isin(codes).to_numpy())


# --- import-time invariants (fail fast on a bad edit) ---
_tested = [c for f in fold_ids() for c in countries_for(f, "test")]
assert sorted(_tested) == sorted(COUNTRY_CODE_TO_NAME), "each country must be tested exactly once"
for _f in fold_ids():
    _tr, _va, _te = (countries_for(_f, r) for r in ("train", "val", "test"))
    assert _tr.isdisjoint(_va) and _tr.isdisjoint(_te) and _va.isdisjoint(_te)
    assert _tr | _va | _te == set(COUNTRY_CODE_TO_NAME)
