import numpy as np
import pandas as pd

from hts_bench.method.naive import Naive, SeasonalNaive


def test_naive_repeats_last_value():
    train = pd.Series([10.0, 20.0, 33.0])
    forecast = Naive().forecast_fit(train).forecast(5, train)
    np.testing.assert_array_equal(forecast, [33.0] * 5)


def test_seasonal_naive_tiles_last_cycle():
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])  # period=4 -> last cycle [5,6,7,8]
    forecast = SeasonalNaive(seasonal_period=4).forecast_fit(train).forecast(4, train)
    np.testing.assert_array_equal(forecast, [5.0, 6.0, 7.0, 8.0])


def test_seasonal_naive_handles_horizon_longer_than_one_period():
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    forecast = SeasonalNaive(seasonal_period=4).forecast_fit(train).forecast(10, train)
    np.testing.assert_array_equal(forecast, [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0])


def test_naive_fitted_values_is_the_one_step_shift():
    train = pd.Series([10.0, 20.0, 33.0])
    fitted = Naive().forecast_fit(train).fitted_values()

    assert fitted.index.equals(train.index)
    assert np.isnan(fitted.iloc[0])
    np.testing.assert_array_equal(fitted.iloc[1:].to_numpy(), [10.0, 20.0])


def test_seasonal_naive_fitted_values_is_the_seasonal_shift():
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fitted = SeasonalNaive(seasonal_period=4).forecast_fit(train).fitted_values()

    assert fitted.iloc[:4].isna().all()
    np.testing.assert_array_equal(fitted.iloc[4:].to_numpy(), [1.0, 2.0, 3.0, 4.0])
