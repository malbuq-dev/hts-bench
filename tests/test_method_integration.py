import os

import numpy as np
import pytest

from hts_bench.data.loader import load_dataset
from hts_bench.method.lgbm_adapter import LGBMAdapter
from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.method.runner import run_forecast
from hts_bench.method.statsmodels_adapter import ARIMA, ETS, Theta

LABOUR_SEASONAL_PERIOD = 12  # monthly
HORIZON = 8  # Nixtla's own suggested horizon for Labour


@pytest.fixture
def labour_dataset():
    if not os.path.exists(os.path.join("dataset", "labour", "meta.json")):
        pytest.skip("dataset/labour not converted yet - run scripts/convert_labour.py")
    return load_dataset("labour")


@pytest.mark.parametrize(
    "method_factory",
    [
        Naive,
        lambda: SeasonalNaive(seasonal_period=LABOUR_SEASONAL_PERIOD),
        lambda: ETS(seasonal_period=LABOUR_SEASONAL_PERIOD),
        ARIMA,
        lambda: Theta(seasonal_period=LABOUR_SEASONAL_PERIOD),
        lambda: LGBMAdapter(n_lags=LABOUR_SEASONAL_PERIOD, verbosity=-1),
    ],
    ids=["Naive", "SeasonalNaive", "ETS", "ARIMA", "Theta", "LightGBM"],
)
def test_method_runs_end_to_end_on_labour(labour_dataset, method_factory):
    forecasts = run_forecast(labour_dataset, method_factory=method_factory, horizon=HORIZON)

    assert forecasts.shape == (HORIZON, len(labour_dataset.bottom_series))
    assert not forecasts.isna().to_numpy().any()
    assert np.isfinite(forecasts.to_numpy()).all()
