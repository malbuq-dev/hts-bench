import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hts_bench.data.loader import load_dataset 
from hts_bench.reconciliation.reconcile import bottom_up, min_trace, min_trace_shrink, top_down
from hts_bench.method.lgbm_adapter import LGBMAdapter 
from hts_bench.method.naive import Naive, SeasonalNaive
from hts_bench.method.statsmodels_adapter import ARIMA, ETS, Theta
from hts_bench.pipeline import run_benchmark

# name: (horizon, seasonal_period, n_origins)
DATASETS = {
    "labour": (8, 12, 5),
    "tourism": (24, 12, 5),
    "traffic": (1, 7, 10),
    "wiki2": (1, 7, 10),
    "m5": (28, 7, None),
}

SKIP_SHRINK_FOR = {"m5"}
PLAIN_RECONCILE_CHOICES = {"bottom_up": bottom_up, "top_down": top_down, "min_trace": min_trace}
ALL_LEVELS_CHOICES = {"top_down", "min_trace", "min_trace_shrink"}


def method_factories(seasonal_period):
    return {
        "naive": Naive,
        "seasonal_naive": lambda: SeasonalNaive(seasonal_period=seasonal_period),
        "ets": lambda: ETS(seasonal_period=seasonal_period),
        "arima": ARIMA,
        "theta": lambda: Theta(seasonal_period=seasonal_period),
        "lightgbm": lambda: LGBMAdapter(n_lags=seasonal_period, verbosity=-1),
    }


def main():
    for dataset_name, (horizon, seasonal_period, n_origins) in DATASETS.items():
        ds = load_dataset(dataset_name)
        all_levels = ds.summing_matrix.row_ids
        factories = method_factories(seasonal_period)

        reconcile_names = list(PLAIN_RECONCILE_CHOICES) + (
            [] if dataset_name in SKIP_SHRINK_FOR else ["min_trace_shrink"]
        )

        for reconcile_name in reconcile_names:
            start = time.time()
            if reconcile_name == "min_trace_shrink":
                reconcile_fn, setup_seconds = min_trace_shrink(
                    ds, lambda: ETS(seasonal_period=seasonal_period), horizon
                )
                # Setup cost is shared across every method run_benchmark loops over
                # below, not a per-method quantity - reported once here rather than
                # folded into the per-method results table (see min_trace_shrink's
                # docstring and evaluate()'s reconcile_seconds for why).
                print(f"[{dataset_name}/{reconcile_name}] covariance setup: {setup_seconds:.2f}s")
            else:
                reconcile_fn = PLAIN_RECONCILE_CHOICES[reconcile_name]

            series_ids = all_levels if reconcile_name in ALL_LEVELS_CHOICES else None
            records_dir = os.path.join("result", dataset_name, reconcile_name)

            run_benchmark(
                dataset_name,
                factories,
                horizon,
                n_origins=n_origins,
                reconcile_fn=reconcile_fn,
                series_ids=series_ids,
                records_dir=records_dir,
            )
            print(f"[{dataset_name}/{reconcile_name}] done in {time.time() - start:.1f}s -> {records_dir}")


if __name__ == "__main__":
    main()
