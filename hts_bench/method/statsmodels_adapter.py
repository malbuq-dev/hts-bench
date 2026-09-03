import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from hts_bench.method.base import MethodBase


class StatsmodelsAdapter(MethodBase):
    """
    Generic adapter wrapping any statsmodels model class - one adapter class for
    many models, mirroring TFB's DartsModelAdapter pattern
    (ts_benchmark/baselines/darts/darts_models.py) instead of writing a near-
    duplicate class per model.
    """

    def __init__(self, model_cls, model_kwargs: dict, method_name: str):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self._name = method_name
        self._result = None

    def forecast_fit(self, train_data: pd.Series) -> "StatsmodelsAdapter":
        model = self.model_cls(train_data, **self.model_kwargs)
        self._result = model.fit()
        return self

    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        return np.asarray(self._result.forecast(horizon))

    def fitted_values(self) -> pd.Series:
        try:
            return self._result.fittedvalues
        except AttributeError:
            # ThetaModelResults has no .fittedvalues - statsmodels doesn't expose
            # an in-sample fit for Theta the way it does for ETS/ARIMA. Not
            # approximated here; a caller that needs residuals from every method
            # (MinT(shrink)) should pick a different method for the levels it
            # fits, not silently get something Theta doesn't actually support.
            raise NotImplementedError(
                f"{self.name} does not implement fitted_values - "
                f"{self.model_cls.__name__} results have no .fittedvalues"
            )

    @property
    def name(self) -> str:
        return self._name


def ETS(seasonal_period: int) -> StatsmodelsAdapter:
    return StatsmodelsAdapter(
        ExponentialSmoothing,
        {"trend": "add", "seasonal": "add", "seasonal_periods": seasonal_period},
        method_name="ETS",
    )


def ARIMA(order=(1, 1, 1)) -> StatsmodelsAdapter:
    # Fixed order, not auto-selected - order/hyperparameter search (e.g. pmdarima's
    # auto_arima) is future work, not silently approximated here.
    return StatsmodelsAdapter(ARIMAModel, {"order": order}, method_name=f"ARIMA{order}")


def Theta(seasonal_period: int) -> StatsmodelsAdapter:
    # ThetaModelResults.forecast(steps, theta=2) - the generic adapter's
    # `self._result.forecast(horizon)` call lines up with `steps` positionally,
    # so no special-casing needed despite the different constructor kwarg name.
    return StatsmodelsAdapter(ThetaModel, {"period": seasonal_period}, method_name="Theta")
