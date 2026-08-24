import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.data.hierarchy import SummingMatrix


def _read_shard(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


def _load_data(dataset_dir: str, data_files: list) -> pd.DataFrame:
    paths = [os.path.join(dataset_dir, f) for f in data_files]
    if len(paths) == 1:
        shards = [_read_shard(paths[0])]
    else:
        # Multiple shard files: read concurrently, same pattern TFB uses in
        # LocalDataSource.load_series_list (ts_benchmark/data/data_source.py).
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            shards = list(pool.map(_read_shard, paths))
    data = pd.concat(shards, axis=1).sort_index()
    return data


def load_dataset(name: str, root: str = "dataset") -> HierarchicalDataset:
    dataset_dir = os.path.join(root, name)

    with open(os.path.join(dataset_dir, "meta.json")) as f:
        meta = json.load(f)

    data = _load_data(dataset_dir, meta["data_files"])
    series_meta = pd.read_csv(
        os.path.join(dataset_dir, "series_meta.csv"), index_col="series_id"
    )

    n_series = meta["n_series"]
    n_bottom = meta["n_bottom"]
    if len(series_meta) != n_series:
        raise ValueError(
            f"{name}: series_meta.csv has {len(series_meta)} rows, expected {n_series}"
        )
    if int(series_meta["is_bottom"].sum()) != n_bottom:
        raise ValueError(
            f"{name}: series_meta.csv has {int(series_meta['is_bottom'].sum())} "
            f"bottom rows, expected {n_bottom}"
        )
    missing_cols = set(series_meta.index) - set(data.columns)
    if missing_cols:
        raise ValueError(f"{name}: series in series_meta.csv missing from data: {missing_cols}")

    return HierarchicalDataset(
        name=meta["name"],
        freq=meta["freq"],
        data=data,
        series_meta=series_meta,
    )


def aggregate_from_bottom(summing_matrix: SummingMatrix, bottom_data: pd.DataFrame) -> pd.DataFrame:
    """
    Derive every series in the hierarchy from bottom-level data: y_t = S @ b_t.

    Sparse matmul so it scales to hierarchies like M5's (S far too large to be
    dense). A single vectorized call either way - not something to hand-parallelize.
    """
    B = bottom_data[summing_matrix.col_ids].to_numpy()  # (T, m)
    Y = summing_matrix.matrix.dot(B.T).T  # (n, m) @ (m, T) -> (n, T) -> (T, n)
    return pd.DataFrame(Y, index=bottom_data.index, columns=summing_matrix.row_ids)
