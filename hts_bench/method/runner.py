import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.method.base import MethodBase


def run_forecast(
    ds: HierarchicalDataset,
    method_factory: Callable[[], MethodBase],
    horizon: int,
    series_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fits and forecasts every series in `series_ids` independently, using a
    fixed train/test split

    """
    series_ids = series_ids if series_ids is not None else ds.bottom_series
    train = ds.data.iloc[:-horizon]
    test_index = ds.data.index[-horizon:]

    forecasts = {}
    times = {}

    for series_id in series_ids:
        method = method_factory()
        train_series = train[series_id]
        start = time.perf_counter()
        method.forecast_fit(train_series)
        forecasts[series_id] = method.forecast(horizon, train_series)
        times[series_id] = time.perf_counter() - start

    return pd.DataFrame(forecasts, index=test_index), times


def compute_residuals(
    ds: HierarchicalDataset,
    method_factory: Callable[[], MethodBase],
    horizon: int,
    series_ids: Optional[List[str]] = None,
) -> pd.DataFrame:

    series_ids = series_ids if series_ids is not None else ds.bottom_series
    train = ds.data.iloc[:-horizon]

    residuals = {}
    for series_id in series_ids:
        method = method_factory()
        train_series = train[series_id]
        method.forecast_fit(train_series)
        residuals[series_id] = train_series - method.fitted_values()

    return pd.DataFrame(residuals)
