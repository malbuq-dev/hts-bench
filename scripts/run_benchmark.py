import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hts_bench.data.loader import load_dataset  # noqa: E402
from hts_bench.evaluation.reconcile import bottom_up, min_trace, top_down  # noqa: E402
from hts_bench.method.lgbm_adapter import LGBMAdapter  # noqa: E402
from hts_bench.method.naive import Naive, SeasonalNaive  # noqa: E402
from hts_bench.method.statsmodels_adapter import ARIMA, ETS, Theta  # noqa: E402
from hts_bench.pipeline import run_benchmark  # noqa: E402

METHOD_NAMES = ["naive", "seasonal_naive", "ets", "arima", "theta", "lightgbm"]
RECONCILE_CHOICES = {"bottom_up": bottom_up, "top_down": top_down, "min_trace": min_trace}


def build_method_factories(names, seasonal_period, n_lags):
    registry = {
        "naive": lambda: Naive(),
        "seasonal_naive": lambda: SeasonalNaive(seasonal_period=seasonal_period),
        "ets": lambda: ETS(seasonal_period=seasonal_period),
        "arima": lambda: ARIMA(),
        "theta": lambda: Theta(seasonal_period=seasonal_period),
        "lightgbm": lambda: LGBMAdapter(n_lags=n_lags, verbosity=-1),
    }
    return {name: registry[name] for name in names}


def main():
    parser = argparse.ArgumentParser(
        description="Run an hts-bench benchmark from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name under dataset/ (e.g. labour)")
    parser.add_argument(
        "--methods", nargs="+", required=True, choices=METHOD_NAMES, help="Methods to compare"
    )
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon")
    parser.add_argument(
        "--n-origins",
        type=int,
        default=None,
        help="If set, use rolling-origin evaluation with this many origins instead of a single split",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=12,
        help="Seasonal period for seasonal_naive/ets/theta",
    )
    parser.add_argument("--n-lags", type=int, default=None, help="Lag window for lightgbm (default: --seasonal-period)")
    parser.add_argument("--metrics", nargs="+", default=["mae", "rmse", "mase"])
    parser.add_argument(
        "--reconcile",
        choices=list(RECONCILE_CHOICES),
        default="bottom_up",
        help="min_trace needs every hierarchy level forecast independently - implies --all-levels",
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Forecast every hierarchy level independently, not just the bottom (needed for --reconcile min_trace)",
    )
    parser.add_argument("--aggregate", default="mean", help="Aggregation across series for the leaderboard (mean/median/max/...)")
    parser.add_argument("--by-level", action="store_true", help="Break the leaderboard out by hierarchy level")
    parser.add_argument("--save-path", default=None, help="Write the leaderboard CSV here")
    parser.add_argument("--records-dir", default=None, help="Also save the raw per-series comparison table here")

    args = parser.parse_args()

    method_factories = build_method_factories(
        args.methods, args.seasonal_period, args.n_lags or args.seasonal_period
    )
    reconcile_fn = RECONCILE_CHOICES[args.reconcile]

    all_levels = args.all_levels or args.reconcile == "min_trace"
    series_ids = load_dataset(args.dataset).summing_matrix.row_ids if all_levels else None

    result = run_benchmark(
        args.dataset,
        method_factories,
        args.horizon,
        n_origins=args.n_origins,
        metric_names=args.metrics,
        reconcile_fn=reconcile_fn,
        series_ids=series_ids,
        aggregate=args.aggregate,
        by_level=args.by_level,
        save_path=args.save_path,
        records_dir=args.records_dir,
    )

    print(result.to_string())


if __name__ == "__main__":
    main()
