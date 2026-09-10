from typing import List

import pandas as pd


def leaderboard(
    comparison: pd.DataFrame,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    aggregate: str = "mean",
    by_level: bool = False,
) -> pd.DataFrame:
    """
    Collapses a compare_methods()/compare_methods_rolling() table into one
    score per method per metric - TFB's leaderboard idea (report/utils/
    leaderboard.py's get_leaderboard), pivoting a per-series table nobody can
    read at a glance into one aggregate number per model.

    Two-stage aggregation, same structure as TFB's: (1) collapse repeated
    readings of the same (method, series_id) - only compare_methods_rolling's
    tables have these, one per origin - via mean, same as TFB's
    pivot_table(aggfunc=nanmean) collapsing duplicate (model, file) rows
    before its own second stage; a no-op on compare_methods's output, which
    only ever has one reading per (method, series_id). (2) aggregate across
    series_id with `aggregate` (mean/median/... - anything pandas .agg
    accepts). Doing this in two stages rather than one flat groupby matters
    for aggregate="median": a method scored over more origins would otherwise
    be mis-weighted relative to one scored over fewer.

    Narrower than TFB here: no NaN-ratio thresholding/fill_type - TFB
    disqualifies a model from a metric if too many of its per-series results
    are NaN, then fills the rest with the column mean before aggregating.
    MASE's only NaN source (metrics.py's flat-history guard) is rare enough on
    real hierarchies not to need a missing-data policy of its own; pandas'
    .agg skips NaNs by default, so a rare one doesn't poison the score, and a
    metric with zero valid series reports NaN honestly rather than a filled-in
    number.

    by_level=True keeps hierarchy level as a second grouping key instead of
    collapsing every level together - meaningful here in a way TFB has no
    equivalent for, since TFB has no hierarchy concept: whether reconciliation
    helps more at the top than the bottom is exactly the kind of question
    by_level=True exists to answer.
    """
    metric_names = list(metric_names)
    df = comparison.reset_index()

    per_series = df.groupby(["method", "series_id", "level"])[metric_names].mean().reset_index()

    group_cols = ["method", "level"] if by_level else ["method"]
    return per_series.groupby(group_cols)[metric_names].agg(aggregate)
