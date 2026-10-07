from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hts_bench.method.base import MethodBase


class LGBMAdapter(MethodBase):
    """
    Per-series LightGBM on lag features, forecasting recursively.

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
