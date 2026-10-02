"""
Reads back everything scripts/run_experiments.py saved to result/ (no
re-fitting) and exports one aggregated leaderboard row per (dataset,
reconcile, method) as JSON, for docs/leaderboard.html to render client-side.

This is a separate, smaller artifact from the raw per-series records: the
page wants one row per method to filter/sort, not thousands of per-series
rows (result/ itself stays gitignored and local-only - see .gitignore).
"""
import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hts_bench.report.leaderboard import leaderboard  # noqa: E402
from hts_bench.report.recording import load_records  # noqa: E402

DATASETS = ["labour", "tourism", "traffic", "wiki2", "m5"]
RECONCILE_CHOICES = ["bottom_up", "top_down", "min_trace", "min_trace_shrink"]
METRIC_NAMES = ["mae", "rmse", "mase", "time_seconds", "reconcile_seconds"]

OUT_PATH = os.path.join(PROJECT_ROOT, "docs", "data", "leaderboard.json")


def main():
    rows = []
    for dataset in DATASETS:
        for reconcile in RECONCILE_CHOICES:
            path = os.path.join(PROJECT_ROOT, "result", dataset, reconcile)
            if not os.path.isdir(path):
                continue
            records = load_records([path])
            board = leaderboard(records, metric_names=METRIC_NAMES, aggregate="mean")
            for method, metric_values in board.iterrows():
                row = {"dataset": dataset, "reconcile": reconcile, "method": method}
                for name in METRIC_NAMES:
                    value = metric_values[name]
                    row[name] = None if value != value else round(float(value), 6)  # NaN -> null
                rows.append(row)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "metrics": METRIC_NAMES,
        "rows": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
