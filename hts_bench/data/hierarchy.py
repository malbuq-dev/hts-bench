from dataclasses import dataclass

import pandas as pd
from scipy import sparse


@dataclass
class SummingMatrix:
    """
    S: y_t = S @ b_t. Sparse because dense storage doesn't scale (e.g. M5's full
    hierarchy is ~42,840 x 30,490 - over a billion cells dense).
    """

    matrix: sparse.csr_matrix  # shape (n_series, n_bottom), 0/1
    row_ids: list  # all series, row order
    col_ids: list  # bottom series, column order


def build_summing_matrix(series_meta: pd.DataFrame) -> SummingMatrix:
    """
    Derive S purely from series_meta - no separate stored matrix needed.

    """
    dim_cols = [c for c in series_meta.columns if c not in ("level", "is_bottom")]
    series_meta = series_meta.rename_axis("series_id")

    bottom = series_meta[series_meta["is_bottom"]]
    col_ids = bottom.index.tolist()
    col_pos = {sid: i for i, sid in enumerate(col_ids)}
    row_ids = series_meta.index.tolist()
    row_pos = {sid: i for i, sid in enumerate(row_ids)}

    rows, cols = [], []
    for level, level_rows in series_meta.groupby("level", sort=False):
        defining_cols = [c for c in dim_cols if level_rows[c].notna().any()]

        if not defining_cols:
            for agg_id in level_rows.index:
                for bottom_id in col_ids:
                    rows.append(row_pos[agg_id])
                    cols.append(col_pos[bottom_id])
            continue

        matches = bottom.reset_index().merge(
            level_rows.reset_index()[["series_id"] + defining_cols],
            on=defining_cols,
            suffixes=("_bottom", "_agg"),
        )
        for bottom_id, agg_id in zip(matches["series_id_bottom"], matches["series_id_agg"]):
            rows.append(row_pos[agg_id])
            cols.append(col_pos[bottom_id])

    data = [1] * len(rows)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(row_ids), len(col_ids))
    )
    return SummingMatrix(matrix=matrix, row_ids=row_ids, col_ids=col_ids)
