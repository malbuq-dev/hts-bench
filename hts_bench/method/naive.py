import numpy as np
import pandas as pd

from hts_bench.method.base import MethodBase


class Naive(MethodBase):
    """Repeats the last observed training value for every step of the horizon."""

    def __init__(self):
        self._last_value = None

    def forecast_fit(self, train_data: pd.Series) -> "Naive":
        self._last_value = train_data.iloc[-1]
        return self

    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        return np.full(horizon, self._last_value, dtype=float)

    @property
    def name(self) -> str:
        return "Naive"


class SeasonalNaive(MethodBase):
    """Repeats the last full seasonal cycle of training data, tiled to the horizon."""

    def __init__(self, seasonal_period: int):
        self.seasonal_period = seasonal_period
        self._last_season = None

    def forecast_fit(self, train_data: pd.Series) -> "SeasonalNaive":
        self._last_season = train_data.iloc[-self.seasonal_period :].to_numpy(dtype=float)
        return self

    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        reps = int(np.ceil(horizon / self.seasonal_period))
        return np.tile(self._last_season, reps)[:horizon]

    @property
    def name(self) -> str:
        return "SeasonalNaive"
