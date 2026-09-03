from typing import Callable, Dict, List, Optional

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.evaluation.reconcile import bottom_up
from hts_bench.evaluation.runner import evaluate, evaluate_rolling
from hts_bench.method.base import MethodBase
from hts_bench.method.runner import run_forecast


def compare_methods(
    ds: HierarchicalDataset,
    method_factories: Dict[str, Callable[[], MethodBase]],
    horizon: int,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Runs every method in `method_factories` through run_forecast -> evaluate and
    stacks the results into one table indexed by (method, series_id) - mirrors
    TFB's eval_model looping over models to build one combined result_df
    (ts_benchmark/evaluation/evaluate_model.py), minus the ParallelBackend/
    ModelFactory scheduling machinery: evaluate()'s docstring already made this
    call for the single-method case, same reasoning applies looping over methods.

    `reconcile_fn` is shared across all methods (not per-method) - reconciliation
    is a property of how you choose to combine a hierarchy's forecasts, not of
    any one method, so comparing methods under a fixed reconciliation choice is
    the meaningful comparison; to compare reconciliation choices instead, call
    this once per reconcile_fn and compare the resulting tables.
    """
    reconcile_fn = reconcile_fn or bottom_up
    results = []
    for method_name, factory in method_factories.items():
        forecasts = run_forecast(ds, method_factory=factory, horizon=horizon)
        result = evaluate(
            ds, forecasts, horizon=horizon, metric_names=metric_names, reconcile_fn=reconcile_fn
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
) -> pd.DataFrame:
    """
    compare_methods's rolling-origin counterpart: runs every method through
    evaluate_rolling instead of evaluate, and stacks the results into one table
    indexed by (method, origin, series_id) - see evaluate_rolling's docstring
    for why a single fixed split isn't enough on its own.
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
        )
        result = result.reset_index()
        result.insert(0, "method", method_name)
        results.append(result)

    return pd.concat(results, ignore_index=True).set_index(["method", "origin", "series_id"])
