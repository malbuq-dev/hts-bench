import numpy as np

from hts_bench.method.statsmodels_adapter import ARIMA, ETS


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


def test_method_names():
    assert ETS(seasonal_period=4).name == "ETS"
    assert ARIMA(order=(2, 1, 0)).name == "ARIMA(2, 1, 0)"
