import numpy as np
import pandas as pd
import pytest

from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.method.runner import compute_residuals
from hts_bench.method.statsmodels_adapter import Theta


def test_compute_residuals_is_train_minus_fitted():
    # horizon=1 holds out the last point, so train is exactly [10, 20, 33].
    full = pd.Series([10.0, 20.0, 33.0, 999.0], index=pd.date_range("2020-01-01", periods=4))

    class Toy:
        def forecast_fit(self, train_data):
            self._train_data = train_data
            return self

        def fitted_values(self):
            return self._train_data.shift(1)

    class Ds:
        data = pd.DataFrame({"s1": full})
        bottom_series = ["s1"]

    residuals = compute_residuals(Ds(), Toy, horizon=1, series_ids=["s1"])

    assert len(residuals) == 3
    assert np.isnan(residuals["s1"].iloc[0])
    assert residuals["s1"].iloc[1] == 20.0 - 10.0
    assert residuals["s1"].iloc[2] == 33.0 - 20.0


def test_compute_residuals_on_real_dataset_matches_run_forecast_defaults():
    import os

    from hts_bench.data.loader import load_dataset

    if not os.path.exists(os.path.join("dataset", "labour", "meta.json")):
        pytest.skip("dataset/labour not converted yet - run scripts/convert_labour.py")
    ds = load_dataset("labour")
    horizon = 8

    residuals = compute_residuals(ds, lambda: SeasonalNaive(seasonal_period=12), horizon)

    assert list(residuals.columns) == ds.bottom_series
    assert len(residuals) == len(ds.data) - horizon
    assert residuals.iloc[12:].notna().all().all()  # past the seasonal_period warm-up


def test_compute_residuals_raises_when_method_lacks_fitted_values():
    full = pd.Series(
        10 + 0.1 * np.arange(25) + np.sin(np.arange(25) / 4 * 2 * np.pi),
        index=pd.date_range("2020-01-01", periods=25, freq="D"),
    )

    class Ds:
        data = pd.DataFrame({"s1": full})
        bottom_series = ["s1"]

    with pytest.raises(NotImplementedError):
        compute_residuals(Ds(), lambda: Theta(seasonal_period=4), horizon=1, series_ids=["s1"])
