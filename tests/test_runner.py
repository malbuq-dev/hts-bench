import pandas as pd

from hts_bench.data.loader import load_dataset
from hts_bench.method.naive import Naive
from hts_bench.method.runner import run_forecast


def test_run_forecast_shape_and_index(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    ds = load_dataset("toy", root=root)

    horizon = 1
    forecasts, times = run_forecast(ds, method_factory=Naive, horizon=horizon)

    assert list(forecasts.columns) == ds.bottom_series  # s1, s2 - bottom only, no s0 (Total)
    pd.testing.assert_index_equal(forecasts.index, toy_data.index[-horizon:], check_names=False)
    assert set(times.keys()) == set(ds.bottom_series)
    assert all(t >= 0 for t in times.values())


def test_run_forecast_naive_matches_last_train_value(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    ds = load_dataset("toy", root=root)

    forecasts, _ = run_forecast(ds, method_factory=Naive, horizon=1)

    last_train_values = toy_data.iloc[:-1].iloc[-1]
    assert forecasts.iloc[0]["s1"] == last_train_values["s1"]
    assert forecasts.iloc[0]["s2"] == last_train_values["s2"]
