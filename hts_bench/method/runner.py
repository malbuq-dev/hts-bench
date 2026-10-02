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
    fixed train/test split (last `horizon` points held out - the same split TFB's
    FixedForecast strategy uses; rolling-origin evaluation is an Evaluation-layer
    concern, not needed here).

    A fresh method instance is built per series via `method_factory` (mirroring
    TFB's own model_factory pattern in evaluate_model.py), since a fitted method
    holds per-series state and can't be reused across series.

    series_ids defaults to ds.bottom_series - returning raw, unreconciled
    forecasts for the bottom level only, not the whole hierarchy, same as
    before this parameter existed. Reconciliation (aggregating those bottom
    forecasts up through S) is applied afterwards, in Evaluation.

    Pass ds.summing_matrix.row_ids instead to fit every series in the hierarchy
    independently, aggregates included - needed for reconciliation methods
    (MinT and the rest of the GLS/trace-minimization family) that require
    genuinely independent base forecasts at more than one level to have
    anything to reconcile. If every non-bottom "forecast" is instead just S @ b̂
    derived from bottom forecasts, those methods collapse to bottom_up exactly
    - see reconciliation/reconcile.py's min_trace docstring for why.

    Returns (forecasts, times): `times` maps each series_id to the wall-clock
    seconds spent on that series' forecast_fit + forecast combined. Per series
    rather than one total, so a method's cost can be broken down the same way
    its accuracy already is (e.g. by hierarchy level) - see evaluation/
    runner.py's evaluate(), which attaches this alongside the accuracy metrics
    for series that were actually fit here (reconciled/aggregate series that
    were instead derived via S @ b̂ have no entry, since nothing was fit for
    them - see compare.py).
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
    """
    Fits every series in `series_ids` the same way run_forecast does, and
    returns each one's in-sample one-step-ahead residuals (train_data -
    fitted_values), aligned on the training index. NaN wherever a method's
    warm-up window hasn't produced a fitted value yet (see MethodBase.
    fitted_values), and the whole column is missing entirely for a method that
    raises NotImplementedError there (e.g. Theta - see statsmodels_adapter.py).

    The only consumer of this is MinT(shrink)'s covariance estimate
    (reconciliation/reconcile.py's shrinkage_covariance) - nothing else in this
    codebase needs in-sample fit, only forecasts.
    """
    series_ids = series_ids if series_ids is not None else ds.bottom_series
    train = ds.data.iloc[:-horizon]

    residuals = {}
    for series_id in series_ids:
        method = method_factory()
        train_series = train[series_id]
        method.forecast_fit(train_series)
        residuals[series_id] = train_series - method.fitted_values()

    return pd.DataFrame(residuals)
