import os
import time
from typing import List

import pandas as pd


def save_record(comparison: pd.DataFrame, save_dir: str, file_prefix: str) -> str:
    """
    Saves a compare_methods()/compare_methods_rolling() table to disk as a
    record - TFB's recording.save_log: raw per-run results written to disk
    separately from the leaderboard, so a leaderboard can be regenerated later
    (different metrics, different aggregate, by_level toggled) via
    report.report()/load_records without re-running any experiment. That's the
    actual point of this module - run_benchmark's save_path alone only ever
    persisted the *final* leaderboard, which loses the per-series detail
    needed to re-aggregate differently.

    Narrower than TFB's version: no compression (TFB's write_record_file
    supports gzip; this project's tables are small enough not to need it),
    and the filename suffix is just a timestamp, not TFB's hostname+pid
    (get_unique_file_suffix - that exists to avoid collisions between
    concurrent worker processes writing to a shared directory; this project
    has no parallel backend, so there's nothing to collide, see evaluation's
    docstrings on why ParallelBackend was skipped).
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_prefix}.{int(time.time())}.csv")
    comparison.to_csv(file_path)
    return file_path


def load_records(paths: List[str]) -> pd.DataFrame:
    """
    Loads one or more saved records back into a single table - TFB's
    recording.load_record_data. Each path can be a file or a directory (every
    *.csv file directly in it is loaded, non-recursively - TFB's
    find_record_files also walks subdirectories; this project's runs don't
    nest results deep enough to need that).

    The returned table is flat (method/series_id/etc as plain columns, not a
    MultiIndex - a CSV round-trip loses index structure anyway), which is
    exactly what report.leaderboard()/leaderboard() expects: it calls
    .reset_index() on its input as the first step regardless, so a table
    that's already flat works unchanged.
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(
                os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".csv")
            )
        else:
            files.append(path)

    if not files:
        raise ValueError(f"no record files found in {paths}")

    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
