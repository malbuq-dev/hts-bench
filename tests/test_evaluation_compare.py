import os

import pytest

from hts_bench.data.loader import load_dataset
from hts_bench.evaluation.compare import compare_methods, compare_methods_rolling
from hts_bench.reconciliation.reconcile import bottom_up, min_trace, top_down
from hts_bench.method.naive import Naive, SeasonalNaive


@pytest.fixture
def toy_ds(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    return load_dataset("toy", root=root)


def test_compare_methods_stacks_one_result_per_method_and_series(toy_ds):
    result = compare_methods(
        toy_ds,
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=1)},
        horizon=1,
    )

    assert result.index.names == ["method", "series_id"]
    assert set(result.index.get_level_values("method")) == {"Naive", "SeasonalNaive"}
    assert set(result.index.get_level_values("series_id")) == {"s0", "s1", "s2"}
    assert len(result) == 2 * 3  # 2 methods x 3 series


def test_compare_methods_uses_a_shared_reconcile_fn(toy_ds):
    bu = compare_methods(toy_ds, {"Naive": Naive}, horizon=1, reconcile_fn=bottom_up)
    td = compare_methods(toy_ds, {"Naive": Naive}, horizon=1, reconcile_fn=top_down)

    # Root total identical either way (see reconcile.py), split differs.
    assert bu.loc[("Naive", "s0"), "mae"] == pytest.approx(td.loc[("Naive", "s0"), "mae"])
    assert bu.loc[("Naive", "s1"), "mae"] != pytest.approx(td.loc[("Naive", "s1"), "mae"])


@pytest.mark.parametrize("name", ["labour", "tourism"])
def test_compare_methods_runs_end_to_end_on_real_datasets(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    result = compare_methods(
        ds,
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)},
        horizon=4,
    )

    assert len(result) == 2 * len(ds.summing_matrix.row_ids)
    assert not result[["mae", "rmse"]].isna().to_numpy().any()


@pytest.mark.parametrize("name", ["labour", "tourism"])
def test_compare_methods_rolling_runs_end_to_end_on_real_datasets(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    n_origins = 3
    result = compare_methods_rolling(
        ds,
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)},
        horizon=4,
        n_origins=n_origins,
    )

    assert result.index.names == ["method", "origin", "series_id"]
    assert len(result) == 2 * n_origins * len(ds.summing_matrix.row_ids)
    assert result.index.get_level_values("origin").nunique() == n_origins
    assert not result[["mae", "rmse"]].isna().to_numpy().any()


def test_compare_methods_series_ids_enables_min_trace(toy_ds):
    # Without series_ids=row_ids, run_forecast only produces bottom-level
    # forecasts and min_trace rejects them outright (see its docstring) -
    # this was a real gap: compare_methods couldn't use min_trace at all
    # before series_ids existed.
    result = compare_methods(
        toy_ds,
        {"Naive": Naive},
        horizon=1,
        reconcile_fn=min_trace,
        series_ids=toy_ds.summing_matrix.row_ids,
    )
    assert set(result.index.get_level_values("series_id")) == {"s0", "s1", "s2"}


def test_compare_methods_rolling_series_ids_enables_min_trace(toy_ds):
    result = compare_methods_rolling(
        toy_ds,
        {"Naive": Naive},
        horizon=1,
        n_origins=1,
        reconcile_fn=min_trace,
        series_ids=toy_ds.summing_matrix.row_ids,
    )
    assert set(result.index.get_level_values("series_id")) == {"s0", "s1", "s2"}
