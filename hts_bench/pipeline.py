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
    Wires Data/Method/Evaluation/Report together the way TFB's own top-level
    pipeline.pipeline() does, minus TFB's config-file/parallel-scheduling layer
    around it: this project's whole benchmark loop runs comfortably in-process,
    so there's nothing to schedule.

    n_origins=None (default) uses compare_methods's single fixed split;
    passing a number switches to compare_methods_rolling instead - see
    evaluate_rolling's docstring for why that matters for the results chapter.

    reconcile_fn defaults to bottom_up (compare_methods's own default).
    For min_trace/min_trace_shrink, pass series_ids explicitly - typically
    load_dataset(dataset_name).summing_matrix.row_ids, which means loading the
    dataset once yourself first if you need this (min_trace_shrink also needs
    a `ds` to be built against, being a reconcile_fn builder itself - see its
    docstring); this function doesn't do that loading twice for you.

    save_path: if given, writes the *leaderboard* (the aggregated table this
    function returns) to this CSV path.

    records_dir: if given, writes the *raw per-series comparison table* (the
    finer-grained data behind the leaderboard, before aggregation) to this
    directory via report.recording.save_record - TFB's pipeline() always does
    this (recording.save_log, unconditionally, for every model). Opt-in here
    instead of mandatory, unlike TFB, to avoid every casual call writing files
    - the point of doing it at all is the same as TFB's: report.report() can
    regenerate a leaderboard from these later (different metrics, different
    aggregate, by_level toggled) without re-running any model. TFB saves one
    record file per model (its eval_model runs per model); this saves one file
    per run instead, since compare_methods already stacks every method into
    one table before this function ever sees it.

    Returns the leaderboard either way.
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
