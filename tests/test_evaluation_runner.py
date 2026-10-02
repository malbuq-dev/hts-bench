import os

import numpy as np
import pandas as pd
import pytest

from hts_bench.data.loader import load_dataset
from hts_bench.reconciliation.reconcile import bottom_up, top_down
from hts_bench.evaluation.runner import evaluate, evaluate_rolling
from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.method.runner import run_forecast


@pytest.fixture
def toy_ds(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    return load_dataset("toy", root=root)


@pytest.fixture
def toy_forecasts(toy_ds):
    return pd.DataFrame({"s1": [60.0], "s2": [20.0]}, index=toy_ds.data.index[-1:])


def test_evaluate_scores_every_series_in_the_hierarchy(toy_ds, toy_forecasts):
    result = evaluate(toy_ds, toy_forecasts, horizon=1)

    assert set(result.index) == {"s0", "s1", "s2"}
    assert list(result.columns) == ["level", "mae", "rmse", "mase"]
    assert not result[["mae", "rmse"]].isna().to_numpy().any()


def test_evaluate_adds_time_seconds_when_times_given(toy_ds, toy_forecasts):
    times = {"s1": 0.01, "s2": 0.02}  # no "s0" - it's never independently fit

    result = evaluate(toy_ds, toy_forecasts, horizon=1, times=times)

    assert list(result.columns) == ["level", "mae", "rmse", "mase", "time_seconds"]
    assert result.loc["s1", "time_seconds"] == pytest.approx(0.01)
    assert result.loc["s2", "time_seconds"] == pytest.approx(0.02)
    assert np.isnan(result.loc["s0", "time_seconds"])  # reconciled, not fit - see docstring


def test_evaluate_reconcile_fn_changes_bottom_level_scores(toy_ds, toy_forecasts):
    bu = evaluate(toy_ds, toy_forecasts, horizon=1, reconcile_fn=bottom_up)
    td = evaluate(toy_ds, toy_forecasts, horizon=1, reconcile_fn=top_down)

    # Root total is identical either way (see reconcile.py), but the split isn't.
    assert bu.loc["s0", "mae"] == pytest.approx(td.loc["s0", "mae"])
    assert bu.loc["s1", "mae"] != pytest.approx(td.loc["s1", "mae"])


def test_evaluate_propagates_nan_for_flat_history_instead_of_crashing(
    write_dataset, toy_series_meta
):
    flat = pd.DataFrame(
        {"s0": [80.0] * 5, "s1": [50.0] * 5, "s2": [30.0] * 5},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    flat.index.name = "date"
    root = write_dataset("flat", flat, toy_series_meta)
    ds = load_dataset("flat", root=root)
    forecasts = pd.DataFrame({"s1": [55.0], "s2": [25.0]}, index=flat.index[-1:])

    result = evaluate(ds, forecasts, horizon=1)

    assert result["mase"].isna().all()  # every series' history is flat
    assert not result[["mae", "rmse"]].isna().to_numpy().any()  # unaffected metrics still compute


@pytest.mark.parametrize("name", ["labour", "tourism", "m5"])
def test_evaluate_runs_end_to_end_on_real_datasets(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    horizon = 4
    seasonal_period = 7 if name == "m5" else 12  # m5 is daily, labour/tourism are monthly

    forecasts, _ = run_forecast(
        ds, method_factory=lambda: SeasonalNaive(seasonal_period=seasonal_period), horizon=horizon
    )
    result = evaluate(ds, forecasts, horizon=horizon)

    assert len(result) == len(ds.summing_matrix.row_ids)
    assert np.isfinite(result[["mae", "rmse"]].to_numpy()).all()


@pytest.fixture
def toy_ds_10days(write_dataset, toy_series_meta):
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    north = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0]
    south = [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0]
    data = pd.DataFrame({"s0": [n + s for n, s in zip(north, south)], "s1": north, "s2": south}, index=dates)
    data.index.name = "date"
    root = write_dataset("toy10", data, toy_series_meta)
    return load_dataset("toy10", root=root)


def test_evaluate_rolling_produces_one_block_per_origin(toy_ds_10days):
    # 10 points, horizon=2, n_origins=3 -> cutoffs at 10, 8, 6 -> origins (first
    # test date of each block) at index 8, 6, 4 -> dates day9, day7, day5.
    result = evaluate_rolling(toy_ds_10days, Naive, horizon=2, n_origins=3)

    origins = sorted(result.index.get_level_values("origin").unique())
    expected = [toy_ds_10days.data.index[i] for i in (4, 6, 8)]
    assert origins == expected
    assert len(result) == 3 * 3 * 1  # 3 origins x 3 series x 1 method


def test_evaluate_rolling_raises_when_not_enough_history(toy_ds_10days):
    with pytest.raises(ValueError, match="n_origins"):
        evaluate_rolling(toy_ds_10days, Naive, horizon=2, n_origins=5)  # 5*2=10 >= 10 points


@pytest.mark.parametrize("name", ["labour", "tourism", "m5"])
def test_evaluate_rolling_runs_end_to_end_on_real_datasets(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    horizon = 4
    seasonal_period = 7 if name == "m5" else 12
    n_origins = 3

    result = evaluate_rolling(
        ds, lambda: SeasonalNaive(seasonal_period=seasonal_period), horizon=horizon, n_origins=n_origins
    )

    assert len(result) == n_origins * len(ds.summing_matrix.row_ids)
    assert result.index.get_level_values("origin").nunique() == n_origins
    assert np.isfinite(result[["mae", "rmse"]].to_numpy()).all()
