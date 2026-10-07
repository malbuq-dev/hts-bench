import time
from typing import Callable, Dict, List, Optional

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.evaluation.metrics import METRICS
from hts_bench.reconciliation.reconcile import bottom_up
from hts_bench.method.base import MethodBase
from hts_bench.method.runner import run_forecast


def evaluate(
    ds: HierarchicalDataset,
    forecasts: pd.DataFrame,
    horizon: int,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
    times: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Scores forecasts against actuals for every series in the hierarchy, one row
    per series indexed by series_id, columns = level + each requested metric
    (+ time_seconds, when `times` is given) + reconcile_seconds.

    """
    reconcile_fn = reconcile_fn or bottom_up
    reconcile_start = time.perf_counter()
    reconciled = reconcile_fn(ds, forecasts)
    reconcile_seconds = time.perf_counter() - reconcile_start

    actuals = ds.data.loc[reconciled.index, reconciled.columns]
    hist = ds.data.iloc[:-horizon]

    rows = []
    for series_id in reconciled.columns:

        row = {
            "series_id": series_id,
            "level": ds.series_meta.loc[series_id, "level"],
        }

        actual = actuals[series_id].to_numpy(dtype=float)
        predicted = reconciled[series_id].to_numpy(dtype=float)
        hist_data = hist[series_id].to_numpy(dtype=float)

        for name in metric_names:
            if name in ("time_seconds", "reconcile_seconds"):
                continue

            row[name] = METRICS[name](actual, predicted, hist_data=hist_data)

        if times is not None:
            row["time_seconds"] = times.get(series_id, float("nan"))

        row["reconcile_seconds"] = reconcile_seconds
        rows.append(row)

    return pd.DataFrame(rows).set_index("series_id")


def evaluate_rolling(
    ds: HierarchicalDataset,
    method_factory: Callable[[], MethodBase],
    horizon: int,
    n_origins: int = 5,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
    series_ids: Optional[List[str]] = None,
) -> pd.DataFrame:

    if n_origins * horizon >= len(ds.data):
        raise ValueError(
            f"n_origins={n_origins} * horizon={horizon} = {n_origins * horizon} points needed, "
            f"but ds.data only has {len(ds.data)}"
        )

    results = []

    for i in range(n_origins):

        cutoff = len(ds.data) - i * horizon

        origin_ds = HierarchicalDataset(
            name=ds.name, freq=ds.freq, data=ds.data.iloc[:cutoff], series_meta=ds.series_meta
        )

        forecasts, times = run_forecast(origin_ds, method_factory, horizon, series_ids=series_ids)
        
        result = evaluate(
            origin_ds, forecasts, horizon, metric_names=metric_names, reconcile_fn=reconcile_fn,
            times=times,
        )

        result = result.reset_index()
        result.insert(0, "origin", origin_ds.data.index[-horizon])
        results.append(result)

    return pd.concat(results, ignore_index=True).set_index(["origin", "series_id"])
