from typing import Callable, Dict, List, Optional

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.reconciliation.reconcile import bottom_up
from hts_bench.evaluation.runner import evaluate, evaluate_rolling
from hts_bench.method.base import MethodBase
from hts_bench.method.runner import run_forecast


def compare_methods(
    ds: HierarchicalDataset,
    method_factories: Dict[str, Callable[[], MethodBase]],
    horizon: int,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
    series_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Runs every method in `method_factories` through run_forecast -> evaluate and
    stacks the results into one table indexed by (method, series_id)

    """
    reconcile_fn = reconcile_fn or bottom_up
    results = []
    for method_name, factory in method_factories.items():
        forecasts, times = run_forecast(ds, method_factory=factory, horizon=horizon, series_ids=series_ids)
        result = evaluate(
            ds, forecasts, horizon=horizon, metric_names=metric_names, reconcile_fn=reconcile_fn,
            times=times,
        )
        result = result.reset_index()
        result.insert(0, "method", method_name)
        results.append(result)

    return pd.concat(results, ignore_index=True).set_index(["method", "series_id"])


def compare_methods_rolling(
    ds: HierarchicalDataset,
    method_factories: Dict[str, Callable[[], MethodBase]],
    horizon: int,
    n_origins: int = 5,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
    series_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    compare_methods's rolling-origin counterpart: runs every method through
    evaluate_rolling instead of evaluate, and stacks the results into one table
    indexed by (method, origin, series_id)

    """
    reconcile_fn = reconcile_fn or bottom_up
    results = []
    for method_name, factory in method_factories.items():
        result = evaluate_rolling(
            ds,
            factory,
            horizon=horizon,
            n_origins=n_origins,
            metric_names=metric_names,
            reconcile_fn=reconcile_fn,
            series_ids=series_ids,
        )
        result = result.reset_index()
        result.insert(0, "method", method_name)
        results.append(result)

    return pd.concat(results, ignore_index=True).set_index(["method", "origin", "series_id"])
