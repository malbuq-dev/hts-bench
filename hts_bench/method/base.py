import abc

import numpy as np
import pandas as pd


class MethodBase(abc.ABC):
    """
    The standard interface for forecasting methods in this platform.

    Deliberately narrower than TFB's ModelBase: univariate only (one bottom-level
    series in, one forecast out). Hierarchy-agnostic - a method never sees the
    hierarchy, aggregates, or S. Reconciliation is an Evaluation-layer concern
    applied afterwards to a whole hierarchy's worth of raw forecasts at once.
    """

    @abc.abstractmethod
    def forecast_fit(self, train_data: pd.Series) -> "MethodBase":
        """
        Fit the method on one series' training data.

        :param train_data: Univariate time series used for training.
        :return: The fitted method object.
        """

    @abc.abstractmethod
    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        """
        Forecast `horizon` steps ahead.

        :param horizon: Forecast length.
        :param series: The series to forecast from - the same data forecast_fit
            saw, for methods (e.g. lookback-window models) that need it again
            explicitly rather than relying on internal state alone.
        :return: A 1-D array of length `horizon`.
        """

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
