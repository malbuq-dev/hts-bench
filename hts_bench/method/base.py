import abc

import numpy as np
import pandas as pd


class MethodBase(abc.ABC):
    """
    The standard interface for forecasting methods in this platform.

    """

    @abc.abstractmethod
    def forecast_fit(self, train_data: pd.Series) -> "MethodBase":
        pass

    @abc.abstractmethod
    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        pass
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the method."""

    def fitted_values(self) -> pd.Series:
        """
        In-sample one-step-ahead fitted values, aligned to the index of the
        training data forecast_fit was called with. NaN wherever a fitted
        value isn't available yet (e.g. SeasonalNaive's first seasonal_period
        points, or a lookback-window model's first window).

        Optional, unlike forecast_fit/forecast: only needed by reconciliation
        methods that estimate a residual covariance (MinT(shrink) - see
        reconciliation/reconcile.py's shrinkage_covariance). Not every method can
        support it cleanly (e.g. Theta, whose statsmodels result has no
        fitted-values concept) - raise NotImplementedError rather than fake it.
        """
        raise NotImplementedError(f"{self.name} does not implement fitted_values")

    def __repr__(self):
        return self.name
