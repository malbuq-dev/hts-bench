import os
import time
from typing import List

import pandas as pd


def save_record(comparison: pd.DataFrame, save_dir: str, file_prefix: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_prefix}.{int(time.time())}.csv")
    comparison.to_csv(file_path)
    return file_path


def load_records(paths: List[str]) -> pd.DataFrame:
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
