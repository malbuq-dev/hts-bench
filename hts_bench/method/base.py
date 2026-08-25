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

    def __repr__(self):
        return self.name
