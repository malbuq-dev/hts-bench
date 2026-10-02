"""
Reads back everything scripts/run_experiments.py saved to result/ (no
re-fitting) and produces the comparison views discussed for the results
chapter: method x domain, reconciliation comparison, reconciliation benefit by
hierarchy depth, hierarchy-type grouping, and (views 5-8) the same
cost/domain/structure breakdowns for time_seconds/reconcile_seconds instead of
mase. Run this as many times as you like, with different slicing, without
ever re-running a model.
"""
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hts_bench.data.loader import load_dataset  # noqa: E402
from hts_bench.report.recording import load_records  # noqa: E402

DATASETS = ["labour", "tourism", "traffic", "wiki2", "m5"]
RECONCILE_CHOICES = ["bottom_up", "top_down", "min_trace", "min_trace_shrink"]
HIERARCHY_TYPE = {
    "labour": "crossed",
    "tourism": "crossed",
    "traffic": "tree",
    "wiki2": "tree",
    "m5": "crossed",  # state/store/cat/dept/item are crossed dims, not a single nested chain
}


def compute_depths(dataset_name: str) -> pd.Series:
    """
    Depth from the bottom for every series in `dataset_name` - 0 at the bottom,
    increasing toward the root - indexed by series_id. Derived from how many of
    series_meta's dimension columns are filled in (bottom rows always specify
    every dimension; the root specifies none), which is comparable across
    datasets even though their literal level *names* aren't ("Country/Region"
    vs "Level2" vs "state_id/dept_id" mean nothing next to each other, but
    "2 steps up from the bottom" does).

    Not a claim that depth=2 means the exact same thing in every dataset -
    datasets have different numbers of levels overall, so this doesn't perfectly
    normalize across them. Good enough to see whether reconciliation's effect
    grows the further from the bottom you look, which a single by-level table
    (tied to one dataset's own level names) can't show across datasets at all.
    """
    ds = load_dataset(dataset_name)
    dim_cols = [c for c in ds.series_meta.columns if c not in ("level", "is_bottom")]
    bottom_dim_count = len(dim_cols)  # bottom rows specify every dimension
    specified = ds.series_meta[dim_cols].notna().sum(axis=1)
    return bottom_dim_count - specified


def load_all() -> pd.DataFrame:
    """One row per (dataset, reconcile, method, series_id[, origin]) - every saved run, combined."""
    frames = []
    for dataset in DATASETS:
        existing = [
            r for r in RECONCILE_CHOICES if os.path.isdir(os.path.join("result", dataset, r))
        ]
        if not existing:
            continue
        depths = compute_depths(dataset)

        for reconcile in existing:
            path = os.path.join("result", dataset, reconcile)
            df = load_records([path])
            df.insert(0, "dataset", dataset)
            df.insert(1, "reconcile", reconcile)
            df["depth"] = df["series_id"].map(depths)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    records = load_all()

    print("=== 1. Method x domain (Bottom-Up, mean MASE per dataset) ===")
    bu = records[records["reconcile"] == "bottom_up"]
    pivot = bu.groupby(["dataset", "method"])["mase"].mean().unstack("method")
    print(pivot.round(3))
    print("\nPer-dataset rank (1 = best):")
    print(pivot.rank(axis=1).astype(int))

    print("\n=== 2. Reconciliation comparison (mean MASE across methods, per dataset) ===")
    recon_pivot = records.groupby(["dataset", "reconcile"])["mase"].mean().unstack("reconcile")
    print(recon_pivot.round(3))

    print("\n=== 3. Reconciliation benefit by hierarchy depth (pooled across all datasets) ===")
    print("(0 = bottom, increasing toward the root - see compute_depths' docstring for the caveat)")
    depth_pivot = records.groupby(["depth", "reconcile"])["mase"].mean().unstack("reconcile")
    print(depth_pivot.round(3))

    print("\n=== 4. Hierarchy type: does reconciliation lift differ by structure? ===")
    # Average of per-dataset means, not raw series pooled together - pooling
    # would let Tourism's 555 series drown out Labour's signal (57 series) in
    # the "crossed" group, e.g. hiding Labour's real Top-Down blowup behind
    # Tourism's much larger, much better Top-Down number.
    dataset_means = records.groupby(["dataset", "reconcile"])["mase"].mean().reset_index()
    dataset_means["hierarchy_type"] = dataset_means["dataset"].map(HIERARCHY_TYPE)
    lift = dataset_means.groupby(["hierarchy_type", "reconcile"])["mase"].mean().unstack("reconcile")
    print(lift.round(3))

    print("\n=== 5. Method cost by domain (Bottom-Up, mean time_seconds per dataset) ===")
    # Same bottom_up filter as view 1, for the same reason: holding the
    # reconciliation strategy fixed isolates "which method is slow to fit" from
    # the series-count confound in view 8 below (bottom_up always fits exactly
    # the bottom series, same set for every method).
    print(bu.groupby(["dataset", "method"])["time_seconds"].mean().unstack("method").round(4))

    print("\n=== 6. Reconciliation cost by domain (mean reconcile_seconds per dataset) ===")
    print(records.groupby(["dataset", "reconcile"])["reconcile_seconds"].mean().unstack("reconcile").round(4))
    print(
        "(min_trace_shrink's covariance setup cost isn't included here - it's a one-time "
        "cost shared across every method, not a per-method/per-row quantity. See the "
        "[dataset/min_trace_shrink] covariance setup: ...s lines run_experiments.py printed.)"
    )

    print("\n=== 7. Reconciliation cost by hierarchy structure ===")
    recon_cost_means = records.groupby(["dataset", "reconcile"])["reconcile_seconds"].mean().reset_index()
    recon_cost_means["hierarchy_type"] = recon_cost_means["dataset"].map(HIERARCHY_TYPE)
    print(
        recon_cost_means.groupby(["hierarchy_type", "reconcile"])["reconcile_seconds"]
        .mean().unstack("reconcile").round(4)
    )

    print("\n=== 8. Fit cost (time_seconds) by reconciliation strategy, all methods pooled ===")
    print(
        "CAUTION: not apples-to-apples across strategies. bottom_up/top_down only fit the "
        "bottom level; min_trace/min_trace_shrink fit every level (more series, more total "
        "work), so part of any difference here is series count, not per-series cost - see "
        "view 5 above for a same-series-set comparison instead."
    )
    print(records.groupby(["dataset", "reconcile"])["time_seconds"].mean().unstack("reconcile").round(4))


if __name__ == "__main__":
    main()
