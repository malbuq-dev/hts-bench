import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_series_meta():
    """
    Total
    +-- North
    +-- South
    """
    return pd.DataFrame(
        [
            {"series_id": "s0", "level": "Total", "is_bottom": False, "region": np.nan},
            {"series_id": "s1", "level": "Region", "is_bottom": True, "region": "North"},
            {"series_id": "s2", "level": "Region", "is_bottom": True, "region": "South"},
        ]
    ).set_index("series_id")


@pytest.fixture
def toy_data():
    """North/South time series matching toy_series_meta, Total = their sum."""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    north = [50, 51, 52]
    south = [30, 29, 28]
    total = [n + s for n, s in zip(north, south)]
    df = pd.DataFrame({"s0": total, "s1": north, "s2": south}, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def crossed_series_meta():
    """
    Total
    +-- Region (North, South)
    +-- Region x Type (bottom: North-A, North-B, South-A, South-B)
    """
    rows = [
        {"series_id": "s0", "level": "Total", "is_bottom": False},
        {"series_id": "s1", "level": "Region", "is_bottom": False, "region": "North"},
        {"series_id": "s2", "level": "Region", "is_bottom": False, "region": "South"},
        {"series_id": "s3", "level": "Region/Type", "is_bottom": True, "region": "North", "type": "A"},
        {"series_id": "s4", "level": "Region/Type", "is_bottom": True, "region": "North", "type": "B"},
        {"series_id": "s5", "level": "Region/Type", "is_bottom": True, "region": "South", "type": "A"},
        {"series_id": "s6", "level": "Region/Type", "is_bottom": True, "region": "South", "type": "B"},
    ]
    return pd.DataFrame(rows).set_index("series_id")


@pytest.fixture
def write_dataset(tmp_path):
    """Factory fixture: writes a minimal on-disk dataset, returns its root dir."""
    import json

    def _write(name, data_df, series_meta_df, data_files=None, freq="D", n_series=None, n_bottom=None):
        ds_dir = tmp_path / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        if data_files is None:
            data_df.to_csv(ds_dir / "data.csv")
            file_list = ["data.csv"]
        else:
            for fname, cols in data_files.items():
                data_df[cols].to_csv(ds_dir / fname)
            file_list = list(data_files.keys())

        series_meta_df.to_csv(ds_dir / "series_meta.csv")

        meta = {
            "name": name,
            "freq": freq,
            "horizon_suggested": 1,
            "n_series": n_series if n_series is not None else len(series_meta_df),
            "n_bottom": (
                n_bottom if n_bottom is not None else int(series_meta_df["is_bottom"].sum())
            ),
            "data_files": file_list,
        }
        with open(ds_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        return str(tmp_path)

    return _write
