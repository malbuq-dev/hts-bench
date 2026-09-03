import numpy as np
import pytest

from hts_bench.method.statsmodels_adapter import ARIMA, ETS, Theta


def test_ets_forecast_shape(periodic_series):
    train, horizon = periodic_series.iloc[:-8], 8
    forecast = ETS(seasonal_period=4).forecast_fit(train).forecast(horizon, train)

    assert forecast.shape == (horizon,)
    assert not np.isnan(forecast).any()


def test_arima_forecast_shape(periodic_series):
    train, horizon = periodic_series.iloc[:-8], 8
    forecast = ARIMA().forecast_fit(train).forecast(horizon, train)

    assert forecast.shape == (horizon,)
    assert not np.isnan(forecast).any()


def test_theta_forecast_shape(periodic_series):
    train, horizon = periodic_series.iloc[:-8], 8
    forecast = Theta(seasonal_period=4).forecast_fit(train).forecast(horizon, train)

    assert forecast.shape == (horizon,)
    assert not np.isnan(forecast).any()


def test_method_names():
    assert ETS(seasonal_period=4).name == "ETS"
    assert ARIMA(order=(2, 1, 0)).name == "ARIMA(2, 1, 0)"
    assert Theta(seasonal_period=4).name == "Theta"


def test_ets_and_arima_fitted_values(periodic_series):
    train = periodic_series.iloc[:-8]

    for method in [ETS(seasonal_period=4), ARIMA()]:
        fitted = method.forecast_fit(train).fitted_values()
        assert fitted.index.equals(train.index)
        assert not fitted.isna().any()


def test_theta_fitted_values_is_not_implemented(periodic_series):
    train = periodic_series.iloc[:-8]
    method = Theta(seasonal_period=4).forecast_fit(train)

    with pytest.raises(NotImplementedError):
        method.fitted_values()
