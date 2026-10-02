from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hts_bench.method.base import MethodBase


class LGBMAdapter(MethodBase):
    """
    Per-series LightGBM on lag features, forecasting recursively.

    Not the "global" LightGBM that won M5 - that trains one model jointly
    across every series in a dataset. MethodBase.forecast_fit only ever sees
    one series (runner.run_forecast calls method_factory per series_id), so
    this is a local variant: one LGBMRegressor per series, same as the ARIMA/
    ETS/Theta adapters. A true global model needs a different entry point that
    fits once across ds.get_bottom_data() - future work, not this class.

    random_state defaults to a fixed seed rather than LightGBM's own default
    (unseeded, genuinely random tree-building) - every other method here is
    deterministic given its inputs, so leaving this one non-reproducible would
    mean re-running the same command could silently change reported numbers.
    Pass a different value (or None, LightGBM's default) explicitly to opt out.
    """

    def __init__(self, n_lags: int = 12, random_state: int = 42, **lgbm_kwargs):
        self.n_lags = n_lags
        self.random_state = random_state
        self.lgbm_kwargs = lgbm_kwargs
        self._model = None
        self._history: List[float] = []

    def _lag_frame(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.lib.stride_tricks.sliding_window_view(values[:-1], self.n_lags)
        y = values[self.n_lags :]
        return X, y

    def forecast_fit(self, train_data: pd.Series) -> "LGBMAdapter":
        values = train_data.to_numpy(dtype=float)
        if len(values) <= self.n_lags:
            raise ValueError(
                f"train_data has {len(values)} points, needs > n_lags={self.n_lags}"
            )
        X, y = self._lag_frame(values)
        self._model = LGBMRegressor(random_state=self.random_state, **self.lgbm_kwargs)
        self._model.fit(X, y)
        self._history = values[-self.n_lags :].tolist()
        self._train_index = train_data.index
        self._train_X = X
        return self

    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        history = list(self._history)
        preds = []
        for _ in range(horizon):
            x = np.asarray(history[-self.n_lags :]).reshape(1, -1)
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)

    def fitted_values(self) -> pd.Series:
        fitted = pd.Series(np.nan, index=self._train_index)
        fitted.iloc[self.n_lags :] = self._model.predict(self._train_X)
        return fitted

    @property
    def name(self) -> str:
        return "LightGBM"
