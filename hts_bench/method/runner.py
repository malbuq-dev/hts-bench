from typing import Callable

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.method.base import MethodBase


def run_forecast(
    ds: HierarchicalDataset, method_factory: Callable[[], MethodBase], horizon: int
) -> pd.DataFrame:
    """
    Fits and forecasts every bottom-level series in `ds` independently, using a
    fixed train/test split (last `horizon` points held out - the same split TFB's
    FixedForecast strategy uses; rolling-origin evaluation is an Evaluation-layer
    concern, not needed here).

    A fresh method instance is built per series via `method_factory` (mirroring
    TFB's own model_factory pattern in evaluate_model.py), since a fitted method
    holds per-series state and can't be reused across series.

    Returns raw, unreconciled forecasts only - one column per bottom series, not
    the whole hierarchy. Reconciliation is applied afterwards, in Evaluation.
    """
    bottom = ds.get_bottom_data()
    train = bottom.iloc[:-horizon]
    test_index = bottom.index[-horizon:]

    forecasts = {}
    for series_id in ds.bottom_series:
        method = method_factory()
        train_series = train[series_id]
        method.forecast_fit(train_series)
        forecasts[series_id] = method.forecast(horizon, train_series)

    return pd.DataFrame(forecasts, index=test_index)
