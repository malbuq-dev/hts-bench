from typing import Callable, Dict, List, Optional

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.data.loader import load_dataset
from hts_bench.evaluation.compare import compare_methods, compare_methods_rolling
from hts_bench.method.base import MethodBase
from hts_bench.report.leaderboard import leaderboard
from hts_bench.report.recording import save_record


def run_benchmark(
    dataset_name: str,
    method_factories: Dict[str, Callable[[], MethodBase]],
    horizon: int,
    n_origins: Optional[int] = None,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    reconcile_fn: Optional[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame]] = None,
    series_ids: Optional[List[str]] = None,
    aggregate: str = "mean",
    by_level: bool = False,
    save_path: Optional[str] = None,
    records_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    One call end to end: load_dataset -> compare_methods(_rolling) -> leaderboard
    - the chain every experiment in this project has been assembling by hand.
    Wires Data/Method/Reconciliation/Evaluation/Report together.

    """

    ds = load_dataset(dataset_name)

    if n_origins is None:
        comparison = compare_methods(
            ds,
            method_factories,
            horizon,
            metric_names=metric_names,
            reconcile_fn=reconcile_fn,
            series_ids=series_ids,
        )
    else:
        comparison = compare_methods_rolling(
            ds,
            method_factories,
            horizon,
            n_origins=n_origins,
            metric_names=metric_names,
            reconcile_fn=reconcile_fn,
            series_ids=series_ids,
        )

    if records_dir is not None:
        save_record(comparison, records_dir, file_prefix=dataset_name)

    result = leaderboard(
        comparison, metric_names=metric_names, aggregate=aggregate, by_level=by_level
    )

    if save_path is not None:
        result.to_csv(save_path)

    return result
