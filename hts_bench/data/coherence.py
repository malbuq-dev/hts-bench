import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.data.loader import aggregate_from_bottom


def check_coherence(ds: HierarchicalDataset, atol: float = 1e-6) -> pd.DataFrame:
    """
    Verify y_t = S @ b_t holds for every series/timestamp in ds.data.

    Returns a long-form DataFrame of violations (empty if the dataset is coherent),
    with columns: date, series_id, actual, reconstructed, diff.
    """
    reconstructed = aggregate_from_bottom(ds.summing_matrix, ds.get_bottom_data())
    actual = ds.data[ds.summing_matrix.row_ids]

    diff = (reconstructed - actual).abs()
    mask = diff > atol

    if not mask.to_numpy().any():
        return pd.DataFrame(columns=["date", "series_id", "actual", "reconstructed", "diff"])

    violations = diff[mask].stack()
    rows = []
    for (date, series_id), d in violations.items():
        rows.append(
            {
                "date": date,
                "series_id": series_id,
                "actual": actual.loc[date, series_id],
                "reconstructed": reconstructed.loc[date, series_id],
                "diff": d,
            }
        )
    return pd.DataFrame(rows)
