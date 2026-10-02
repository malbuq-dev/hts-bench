from typing import Callable, List, Optional

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
) -> pd.DataFrame:
    """
    Scores forecasts against actuals for every series in the hierarchy, one row
    per series indexed by series_id, columns = level + each requested metric.

    Mirrors TFB's Evaluator/FixedForecast split (ts_benchmark/evaluation/
    evaluator.py, .../strategy/fixed_forecast.py): metrics are pure functions
    keyed by name (METRICS), and each series gets one result row built from a
    train/test split. Deliberately narrower than TFB here too, same spirit as
    MethodBase/runner.run_forecast - no per-model timing, scaler, or parallel-
    backend machinery, since this benchmark runs single-machine at a scale that
    doesn't need it.

    Where this *isn't* narrower than TFB: TFB scores one flat set of series with
    no hierarchy concept. `forecasts` here holds raw bottom-level forecasts only
    (runner.run_forecast's output) - reconcile_fn (default: bottom_up) sums them
    up through S first, so every aggregate level gets scored too, not just the
    bottom. That reconciliation step has no TFB analogue at all.
    """
    reconcile_fn = reconcile_fn or bottom_up
    reconciled = reconcile_fn(ds, forecasts)

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
            row[name] = METRICS[name](actual, predicted, hist_data=hist_data)
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
    """
    Repeats the single-split run_forecast -> evaluate cycle at n_origins cutoffs,
    stepping back through ds.data by `horizon` each time, and stacks the results
    with an extra `origin` column (the date each run's test window starts at).
    evaluate()'s single fixed split reports accuracy for one particular
    train/test boundary and says nothing about how much that number would move
    under a different one - this is the standard rolling-origin/time-series-CV
    fix (what M4/M5 and R's fable default to). To compare methods this way, see
    compare.compare_methods_rolling.

    Origins are non-overlapping test blocks (each exactly `horizon` long, one
    right after another going back in time), not a finer-grained sliding
    window with its own step size - simpler to reason about, and enough origins
    to show whether results are stable without multiplying runtime by more than
    n_origins. Aggregate across origins yourself, e.g.
    result.groupby(["method", "series_id"])[["mae", "rmse", "mase"]].agg(["mean", "std"]).

    Reuses run_forecast/evaluate unchanged: each origin gets its own
    HierarchicalDataset with `data` truncated to that origin's cutoff (same
    series_meta, so the same hierarchy) - not a change to run_forecast itself,
    which stays a single fixed split.

    series_ids is forwarded to every origin's run_forecast call unchanged
    (default: bottom-only) - the same set of series_ids is valid at every
    origin, since it's derived from series_meta, not from how much data is
    truncated. Pass ds.summing_matrix.row_ids when `reconcile_fn` is
    min_trace/min_trace_shrink, same as compare_methods.
    """
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

        forecasts = run_forecast(origin_ds, method_factory, horizon, series_ids=series_ids)
        result = evaluate(
            origin_ds, forecasts, horizon, metric_names=metric_names, reconcile_fn=reconcile_fn
        )
        result = result.reset_index()
        result.insert(0, "origin", origin_ds.data.index[-horizon])
        results.append(result)

    return pd.concat(results, ignore_index=True).set_index(["origin", "series_id"])
