from typing import List

import pandas as pd


def leaderboard(
    comparison: pd.DataFrame,
    metric_names: List[str] = ("mae", "rmse", "mase"),
    aggregate: str = "mean",
    by_level: bool = False,
) -> pd.DataFrame:

    metric_names = list(metric_names)
    df = comparison.reset_index()

    per_series = df.groupby(["method", "series_id", "level"])[metric_names].mean().reset_index()

    group_cols = ["method", "level"] if by_level else ["method"]
    return per_series.groupby(group_cols)[metric_names].agg(aggregate)
