import numpy as np
import pandas as pd
import pytest
from poverty_cnn.data import splits


def test_all_23_countries_mapped():
    assert len(splits.COUNTRY_CODE_TO_NAME) == 23
    assert len(set(splits.COUNTRY_CODE_TO_NAME.values())) == 23


def test_each_country_tested_exactly_once():
    tested = [c for f in splits.fold_ids() for c in splits.countries_for(f, "test")]
    assert sorted(tested) == sorted(splits.COUNTRY_CODE_TO_NAME)  # 23, no repeats


def test_roles_disjoint_and_cover_within_fold():
    for f in splits.fold_ids():
        tr = splits.countries_for(f, "train")
        va = splits.countries_for(f, "val")
        te = splits.countries_for(f, "test")
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert tr | va | te == set(splits.COUNTRY_CODE_TO_NAME)


def test_clusters_for_returns_row_positions():
    meta = pd.DataFrame({"country": ["AO", "KE", "AO", "ZW"]})
    # fold A tests angola(AO); KE/ZW are not in A-test
    rows = splits.clusters_for(meta, "A", "test")
    assert set(rows.tolist()) == {0, 2}


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        splits.countries_for("A", "bogus")
