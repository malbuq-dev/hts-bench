from typing import List, Optional

import pandas as pd

from hts_bench.report.leaderboard import leaderboard
from hts_bench.report.recording import load_records


def report(
    record_paths: List[str],
    metric_names: List[str] = ("mae", "rmse", "mase"),
    aggregate: str = "mean",
    by_level: bool = False,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates a leaderboard from saved records

    record_paths: files and/or directories, same as recording.load_records.
    save_path: if given, writes the leaderboard to this CSV path.
    """
    records = load_records(record_paths)
    result = leaderboard(records, metric_names=metric_names, aggregate=aggregate, by_level=by_level)

    if save_path is not None:
        result.to_csv(save_path)

    return result
