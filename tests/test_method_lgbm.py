import numpy as np
import pandas as pd
import pytest

from hts_bench.method.lgbm_adapter import LGBMAdapter


def test_lgbm_forecast_shape(periodic_series):
    train, horizon = periodic_series.iloc[:-8], 8
    forecast = LGBMAdapter(n_lags=4, verbosity=-1).forecast_fit(train).forecast(horizon, train)

    assert forecast.shape == (horizon,)
    assert not np.isnan(forecast).any()


def test_lgbm_raises_when_train_data_too_short():
    train = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        LGBMAdapter(n_lags=4, verbosity=-1).forecast_fit(train)


def test_lgbm_name():
    assert LGBMAdapter().name == "LightGBM"


def test_lgbm_fitted_values(periodic_series):
    train = periodic_series.iloc[:-8]
    method = LGBMAdapter(n_lags=4, verbosity=-1).forecast_fit(train)
    fitted = method.fitted_values()

    assert fitted.index.equals(train.index)
    assert fitted.iloc[:4].isna().all()
    assert not fitted.iloc[4:].isna().any()
