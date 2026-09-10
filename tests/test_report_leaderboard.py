import os

import pandas as pd
import pytest

from hts_bench.data.loader import load_dataset
from hts_bench.evaluation.compare import compare_methods, compare_methods_rolling
from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.report.leaderboard import leaderboard


@pytest.fixture
def toy_ds(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    return load_dataset("toy", root=root)


def test_leaderboard_averages_across_series():
    comparison = pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "series_id": ["s1", "s2", "s1", "s2"],
            "level": ["bottom", "bottom", "bottom", "bottom"],
            "mae": [10.0, 20.0, 5.0, 5.0],
        }
    ).set_index(["method", "series_id"])

    result = leaderboard(comparison, metric_names=["mae"])

    assert result.loc["A", "mae"] == 15.0
    assert result.loc["B", "mae"] == 5.0


def test_leaderboard_by_level_keeps_levels_separate():
    comparison = pd.DataFrame(
        {
            "method": ["A", "A"],
            "series_id": ["s0", "s1"],
            "level": ["Total", "bottom"],
            "mae": [10.0, 20.0],
        }
    ).set_index(["method", "series_id"])

    result = leaderboard(comparison, metric_names=["mae"], by_level=True)

    assert result.loc[("A", "Total"), "mae"] == 10.0
    assert result.loc[("A", "bottom"), "mae"] == 20.0


def test_leaderboard_averages_rolling_origins_before_averaging_across_series():
    # 3 origins for s1 (values 0,10,20 -> mean 10), 1 origin for s2 (value 100).
    # A flat mean over all 4 rows would give (0+10+20+100)/4=32.5, wrongly
    # letting s1's extra origins dominate; the correct two-stage answer treats
    # both series equally once collapsed: mean(10, 100) = 55.
    comparison = pd.DataFrame(
        {
            "method": ["A"] * 4,
            "series_id": ["s1", "s1", "s1", "s2"],
            "level": ["bottom"] * 4,
            "mae": [0.0, 10.0, 20.0, 100.0],
        }
    ).set_index(["method", "series_id"])

    result = leaderboard(comparison, metric_names=["mae"])

    assert result.loc["A", "mae"] == 55.0


def test_leaderboard_accepts_arbitrary_pandas_aggregate():
    comparison = pd.DataFrame(
        {
            "method": ["A", "A", "A"],
            "series_id": ["s1", "s2", "s3"],
            "level": ["bottom"] * 3,
            "mae": [1.0, 2.0, 100.0],
        }
    ).set_index(["method", "series_id"])

    assert leaderboard(comparison, metric_names=["mae"], aggregate="median").loc["A", "mae"] == 2.0
    assert leaderboard(comparison, metric_names=["mae"], aggregate="max").loc["A", "mae"] == 100.0


def test_leaderboard_runs_end_to_end_on_compare_methods(toy_ds):
    comparison = compare_methods(
        toy_ds, {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=1)}, horizon=1
    )

    result = leaderboard(comparison)

    assert set(result.index) == {"Naive", "SeasonalNaive"}
    assert list(result.columns) == ["mae", "rmse", "mase"]


@pytest.mark.parametrize("name", ["labour", "tourism"])
def test_leaderboard_runs_end_to_end_on_compare_methods_rolling(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    comparison = compare_methods_rolling(
        ds,
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)},
        horizon=4,
        n_origins=3,
    )

    result = leaderboard(comparison, by_level=True)

    assert set(result.index.get_level_values("method")) == {"Naive", "SeasonalNaive"}
    assert not result.isna().to_numpy().any()
