import os

import pytest

from hts_bench.data.coherence import check_coherence
from hts_bench.data.loader import load_dataset


@pytest.mark.parametrize("name", ["labour", "tourism", "m5", "traffic", "wiki2"])
def test_real_datasets_are_coherent(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    violations = check_coherence(ds)
    assert len(violations) == 0, violations.head()


def test_check_coherence_detects_corruption(write_dataset, toy_series_meta, toy_data):
    corrupted = toy_data.copy()
    corrupted.loc[corrupted.index[0], "s0"] += 1000  # break Total = North + South

    root = write_dataset("toy", corrupted, toy_series_meta)
    ds = load_dataset("toy", root=root)

    violations = check_coherence(ds)
    assert len(violations) == 1
    assert violations.iloc[0]["series_id"] == "s0"
