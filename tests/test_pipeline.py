import os

import pandas as pd
import pytest

from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.pipeline import run_benchmark
from hts_bench.report.report import report


@pytest.mark.parametrize("name", ["labour", "tourism"])
def test_run_benchmark_single_split(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")

    result = run_benchmark(
        name, {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)}, horizon=4
    )

    assert set(result.index) == {"Naive", "SeasonalNaive"}
    assert list(result.columns) == ["mae", "rmse", "mase"]
    assert not result.isna().to_numpy().any()


@pytest.mark.parametrize("name", ["labour", "tourism"])
def test_run_benchmark_rolling_and_by_level(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")

    result = run_benchmark(
        name,
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)},
        horizon=4,
        n_origins=3,
        by_level=True,
    )

    assert result.index.names == ["method", "level"]
    assert set(result.index.get_level_values("method")) == {"Naive", "SeasonalNaive"}
    assert not result.isna().to_numpy().any()


def test_run_benchmark_saves_csv_when_save_path_given(tmp_path):
    if not os.path.exists(os.path.join("dataset", "labour", "meta.json")):
        pytest.skip("dataset/labour not converted yet - run scripts/convert_labour.py")

    out_path = tmp_path / "leaderboard.csv"
    result = run_benchmark("labour", {"Naive": Naive}, horizon=4, save_path=str(out_path))

    assert out_path.exists()
    from_disk = pd.read_csv(out_path, index_col=0)
    pd.testing.assert_series_equal(from_disk["mae"], result["mae"], check_names=False)


def test_run_benchmark_records_dir_allows_reaggregating_without_rerunning(tmp_path):
    if not os.path.exists(os.path.join("dataset", "labour", "meta.json")):
        pytest.skip("dataset/labour not converted yet - run scripts/convert_labour.py")

    records_dir = tmp_path / "result"
    mean_result = run_benchmark(
        "labour",
        {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(seasonal_period=12)},
        horizon=4,
        records_dir=str(records_dir),
    )

    assert len(list(records_dir.iterdir())) == 1

    # No model was fit again here - report() only reads back what was saved.
    max_result = report([str(records_dir)], aggregate="max")

    assert (max_result["mae"] >= mean_result["mae"]).all()
