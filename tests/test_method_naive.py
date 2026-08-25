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
