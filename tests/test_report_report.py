import os

import pandas as pd
import pytest

from hts_bench.report.recording import save_record
from hts_bench.report.report import report


@pytest.fixture
def comparison():
    return pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "series_id": ["s1", "s2", "s1", "s2"],
            "level": ["bottom"] * 4,
            "mae": [10.0, 20.0, 5.0, 5.0],
        }
    ).set_index(["method", "series_id"])


def test_report_builds_a_leaderboard_from_saved_records(tmp_path, comparison):
    path = save_record(comparison, str(tmp_path), file_prefix="toy")

    result = report([path], metric_names=["mae"])

    assert result.loc["A", "mae"] == 15.0
    assert result.loc["B", "mae"] == 5.0


def test_report_can_be_regenerated_differently_without_new_records(tmp_path, comparison):
    path = save_record(comparison, str(tmp_path), file_prefix="toy")

    mean_result = report([path], metric_names=["mae"], aggregate="mean")
    max_result = report([path], metric_names=["mae"], aggregate="max")

    assert mean_result.loc["A", "mae"] == 15.0
    assert max_result.loc["A", "mae"] == 20.0


def test_report_saves_to_disk_when_save_path_given(tmp_path, comparison):
    record_path = save_record(comparison, str(tmp_path / "records"), file_prefix="toy")
    out_path = tmp_path / "leaderboard.csv"

    result = report([record_path], metric_names=["mae"], save_path=str(out_path))

    assert os.path.exists(out_path)
    from_disk = pd.read_csv(out_path, index_col=0)
    pd.testing.assert_series_equal(from_disk["mae"], result["mae"], check_names=False)
